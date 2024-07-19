import logging

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from arguments import ModelArguments, DataArguments
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.config = self.hf_model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        #相当于是一个 group_size 个 cls 的多分类任务
        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs) #XLMR本身不带分类头
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        state_dict = self.hf_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)

class CLEncoder(CrossEncoder):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments, pooling_method = 'cls'):
        super(CrossEncoder, self).__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = self.hf_model.config
        self.pooling_method = pooling_method

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        try:
            del hf_model.classifier
        except:
            print("model has no classifier head")

        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker
    
    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d

    def get_embedding(self, input_ids, attention_mask):
        hidden_state = self.hf_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1].cpu()
        attention_mask = attention_mask.cpu()
        embeddings = self.pooling(hidden_state, attention_mask)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings

    def infoNCELoss(self, anchor, positive, negatives, temperature=1):
        # 计算所有样本的相似度
        pos_similarity = F.cosine_similarity(anchor, positive, dim=-1)
        # 将anchor重复到与负样本相同数量的维度，以便计算
        neg_similarity = F.cosine_similarity(anchor, negatives, dim=-1)
        # 合并正样本和负样本的相似度
        # print(pos_similarity.shape)
        # print(neg_similarity.shape)  
        all_similarity = torch.cat([pos_similarity, neg_similarity])
        # 应用温度缩放
        all_similarity /= temperature
        # 计算InfoNCE损失
        loss = - torch.log(torch.exp(pos_similarity)/torch.sum(torch.exp(all_similarity)))
        return loss.mean()

    def batchloss(self, embeddings):
        # 遍历每个batch计算损失
        losses = []
        for i in range(embeddings.size(0)):
            # anchor embeddings
            anchor = embeddings[i, 0].unsqueeze(0)  # [1, 768]
            # positive embeddings
            positive = embeddings[i, 1].unsqueeze(0)  # [1, 768]
            # 除了anchor和positive之外的所有embeddings作为负样本
            negatives = embeddings[i, 2:]  # [len(negs), 768]
            # 计算当前batch的InfoNCE损失
            # print("anchor", anchor.shape)
            # print("positive", positive.shape)
            # print("negatives", negatives.shape)
            loss = self.infoNCELoss(anchor, positive, negatives)
            losses.append(loss)
        # 计算整个batch的平均损失
        batch_loss = torch.mean(torch.stack(losses))
        return batch_loss

    def forward(self, batch):
        embeddings = self.get_embedding(batch["input_ids"], batch["attention_mask"])
        embeddings = embeddings.reshape(self.train_args.per_device_train_batch_size, self.data_args.train_group_size+1, -1)
        # print("embeddings", embeddings.shape)
        loss = self.batchloss(embeddings).cuda()
        #相当于是一个 group_size 个 cls 的多分类任务
        if self.training:
            return SequenceClassifierOutput(
                loss=loss,
                hidden_states=embeddings,
            )
        else:
            return embeddings

# 投影头
class SimpleResBlock(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class CLProjEncoder(CLEncoder):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments):
        super().__init__(hf_model, model_args, data_args, train_args)
        channels = 768
        # self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, batch):
        embeddings = self.get_embedding(**batch)
        embeddings = embeddings.reshape(self.train_args.per_device_train_batch_size, self.data_args.train_group_size+1, -1)
        # 对 query 做一投影
        querys = embeddings[:,0,:]
        embeddings[:,0,:] = self.proj(querys.cuda()).cpu()
        # print("embeddings", embeddings.shape)
        loss = self.batchloss(embeddings).cuda()
        #相当于是一个 group_size 个 cls 的多分类任务
        if self.training:
            return SequenceClassifierOutput(
                loss=loss,
                hidden_states=embeddings,
            )
        else:
            return embeddings