import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.file_utils import ModelOutput


from FlagEmbedding.visual.eva_clip import create_eva_vision_and_transforms
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    c_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class Visualized_BGE(nn.Module):
    def __init__(self,
                 model_name_bge: str = None,
                 model_weight = None, # "/path/to/your/weight/file/"
                 normlized: bool = True,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 0.02, # 1.0
                 from_pretrained=None, # local config file and model 
                 ):
        super().__init__()

        assert model_name_bge in ["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"]
        assert model_weight is not None
        
        self.model_name_bge = model_name_bge
        
        if model_name_bge == 'BAAI/bge-base-en-v1.5':
            model_name_eva = "EVA02-CLIP-B-16"
            self.hidden_dim = 768
            self.depth = 12
        elif model_name_bge == 'BAAI/bge-m3':
            model_name_eva = "EVA02-CLIP-L-14"
            self.hidden_dim = 1024
            self.depth = 24
        
        if not from_pretrained:
            bge_config = AutoConfig.from_pretrained(model_name_bge)
            bge = AutoModel.from_config(bge_config)
        else:
            print("Loading from local path.")
            bge_config = AutoConfig.from_pretrained(from_pretrained, local_files_only=True)
            bge = AutoModel.from_config(bge_config)
        
        self.bge_encoder = bge.encoder
        self.bge_embeddings = bge.embeddings
        self.bge_pooler = bge.pooler

        self.model_visual, self.preprocess_train, self.preprocess_val= create_eva_vision_and_transforms(
            model_name_eva, 
            force_custom_clip=True)

        
        self.visual_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
        self.load_model(model_weight)
        
        if not from_pretrained:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_bge, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained, use_fast=False)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.to(self.device)
    
    def load_model(self, model_weight):
        self.load_state_dict(torch.load(model_weight, map_location='cpu'))
    
    def gradient_checkpointing_enable(self, **kwargs):
        # self.bge_encoder.gradient_checkpointing_enable()
        self.model_visual.set_grad_checkpointing(True)
    
    
    
    def encode(self, image=None, text=None):
        # used for simple inference
        if image is not None:
            image = self.preprocess_val(Image.open(image)).unsqueeze(0)

            if text is not None:
                text = self.tokenizer(text, return_tensors="pt", padding=True)
                return self.encode_mm(image.to(self.device), text.to(self.device))
            else:
                return self.encode_image(image.to(self.device))
        else:
            if text is not None:
                text = self.tokenizer(text, return_tensors="pt", padding=True)
                return self.encode_text(text.to(self.device))
            else:
                return None
        
    
    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = torch.float16
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        
        return extended_attention_mask

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    
    def encode_text(self, texts):
        '''
        encode text only
        '''
        input_ids = texts['input_ids']
        attention_mask = texts['attention_mask']

        input_shape = input_ids.size()
        device = input_ids.device

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        head_mask = [None] * self.depth
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        
        embedding_output = self.bge_embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        encoder_outputs = self.bge_encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        # pooled_output = self.bge_pooler(sequence_output) if self.bge_pooler is not None else None

        t_reps = self.sentence_embedding(sequence_output, texts['attention_mask']) # tensor: reps with pooling
        if self.normlized:
            t_reps = torch.nn.functional.normalize(t_reps, dim=-1)
        return t_reps.contiguous()

    def encode_mm(self, images:torch.Tensor, texts):
        img_token_emb = self.img_token_embedding(images) #[B, Patch_num, C]
        img_token_emb = img_token_emb[:,1:]              # img_cls is not used here
        img_token_emb = self.visual_proj(img_token_emb)
        device = img_token_emb.device
        
        img_token_len = img_token_emb.size()[1]

        # image position embedding, default position: bge_cls + img tokens + texts
        img_token_position_ids = torch.arange(1, 1 + img_token_len).to(device=device)
        img_position_embeddings = self.bge_embeddings.position_embeddings(img_token_position_ids)
        img_token_emb = img_token_emb + img_position_embeddings

        img_token_emb = self.bge_embeddings.LayerNorm(img_token_emb)

        ### deal with prompt/text
        prompt_input_ids = texts['input_ids']
        prompt_attention_mask = texts['attention_mask']
        prom_input_shape = prompt_input_ids.size()
        
        # bert
        batch_size = prom_input_shape[0]
        prompt_len = prom_input_shape[1]
        prompt_start = 1 + img_token_len

        
        cls_id = torch.tensor([0]).to(device=device)
        prompt_position_ids = torch.arange(prompt_start, prompt_start + prompt_len - 1).to(device=device)
        prompt_position_ids = torch.cat([cls_id, prompt_position_ids]).to(device=device)

        prompt_token_type_ids = torch.zeros(prom_input_shape, dtype=torch.long, device=device)
        prompt_embedding_output = self.bge_embeddings(
            input_ids=prompt_input_ids,
            position_ids=prompt_position_ids,
            token_type_ids=prompt_token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )  # [B, T, C]
        
        
        cls_token = prompt_embedding_output[:, 0:1, :] # bge_cls token
        prompt_embedding_output = prompt_embedding_output[:, 1:]

        prompt_img_embedding = torch.cat([cls_token, img_token_emb, prompt_embedding_output], dim=1)
        
        img_attention_mask = torch.ones(batch_size, img_token_len, device=device)  
        prom_img_attention_mask = torch.cat([img_attention_mask, prompt_attention_mask], dim=1)
        prom_img_input_shape = prompt_img_embedding.size()

        head_mask = [None] * self.depth
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(prom_img_attention_mask, prom_img_input_shape).to(prompt_img_embedding.dtype)
        
        
        encoder_outputs = self.bge_encoder(
            prompt_img_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        
        prompt_img_reps = self.sentence_embedding(sequence_output, prom_img_attention_mask) # tensor: reps with pooling
        if self.normlized:
            prompt_img_reps = torch.nn.functional.normalize(prompt_img_reps, dim=-1)
        return prompt_img_reps

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def img_token_embedding(self, images):
        if images is None:
            return None
        img_token_emb = self.model_visual.encode_image(images, normalize=False) # return_all_features=True, [B, Patch_num, C] 
        
        return img_token_emb.contiguous()
    
    def encode_image(self, images):
        if images is None:
            return None
        
        batch_size = images.shape[0]
        prompts = [""] * batch_size
        
        prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)        
        prompts = prompts.to(images.device)
        img_reps = self.encode_mm(images, prompts)
        return img_reps
    
    def forward(self, mm_it_query=None, image_candidate=None, text_candidate=None, text_query=None, mm_it_candidate=None, task_type=None):
        ### for stage-2 training
        if task_type == "edit_image":
            mm_query_reps = self.encode_mm(mm_it_query[0], mm_it_query[1])
            image_candi_reps = self.encode_image(image_candidate)
            query_reps = mm_query_reps
            candi_reps = image_candi_reps
                
        elif task_type == "t2it":
            text_query_reps = self.encode_text(text_query)
            mmit_candi_reps = self.encode_mm(mm_it_candidate[0], mm_it_candidate[1])
            query_reps = text_query_reps
            candi_reps = mmit_candi_reps
            
        
        if self.training:
            if self.negatives_cross_device:
                query_reps = self._dist_gather_tensor(query_reps)
                candi_reps = self._dist_gather_tensor(candi_reps)

            scores = self.compute_similarity(query_reps, candi_reps)
            scores = scores / self.temperature
            scores = scores.view(query_reps.size(0), -1)
            
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (candi_reps.size(0) // query_reps.size(0))
            
            loss_edit = self.compute_loss(scores, target)
            loss = loss_edit
            
            logging.info("task types: %s; loss: %s" %(task_type, str(loss_edit)))
        else:
            scores = self.compute_similarity(query_reps, candi_reps)
            loss=None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=query_reps,
            c_reps=candi_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        torch.save(self.state_dict(), os.path.join(output_dir, 'Visualized_BGE.pth'))
