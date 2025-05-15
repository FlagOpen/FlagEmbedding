from typing import cast, List, Union, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, CLIPModel, CLIPImageProcessor, CLIPTokenizer
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn
from flag_dataset import MMIT_Dataset, MMIT_Collator, Image_Dataset, Image_Collator
class Flag_mmret(nn.Module):
    def __init__(
            self,
            model_name: str = None,
            normlized: bool = True,
            pooling_method: str = 'cls',
            use_fp16: bool=True,
            image_dir: str = None,
    ) -> None:
        super().__init__()
        
        self.model = AutoModel.from_pretrained(model_name)        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)


        self.normalize_embeddings = normlized
        self.pooling_method = pooling_method

        self.image_dir = image_dir

        if use_fp16: 
            self.use_fp16 = True
            self.model.half()
        else:
            self.use_fp16 = False
            
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)


    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int=256,
                       max_length: int=77,
                       query_type: str = None,
                       ) -> np.ndarray:
       
        
        if query_type == 'text':        
            input_texts = queries
            
            return self.encode_text(input_texts, batch_size=batch_size, max_length=max_length)
        elif query_type == 'mm_it':
            q_text, q_img = queries
            
            input_texts = q_text
            
            
            return self.encode_mm_it(input_texts, q_img, batch_size=batch_size)
        elif query_type == 'image':
            q_img = queries
            return self.encode_image(q_img, batch_size=batch_size)
        else:
            raise NotImplementedError


    def encode_corpus(self,
                      corpus: dict,
                      batch_size: int=256,
                      max_length: int=77,
                      corpus_type: str = None,
                      ) -> np.ndarray:
        if corpus_type == 'text':
            return self.encode_text(corpus["text"], batch_size=batch_size, max_length=max_length)
        elif corpus_type == 'mm_it':
            return self.encode_mm_it(corpus["text"], corpus["image"], batch_size=batch_size, max_length=max_length)
        elif corpus_type == 'image':
            return self.encode_image(corpus["image"], batch_size=batch_size, max_length=max_length)
        else:
            raise RuntimeError(f"You must choose a corpus type from: [mm_it, text, image]")
        


    @torch.no_grad()
    def encode_text(self, sentences: Union[List[str], str], batch_size: int=256, max_length: int=77) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings", disable=len(sentences)<256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            
            embeddings = self.model.get_text_features(**inputs)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


    @torch.no_grad()
    def encode_mm_it(self, captions: Union[List[str], str], image_ids: Union[List[str], str],  batch_size: int=256, max_length: int=77) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(captions, str):
            captions = [captions]
            image_ids = [image_ids]
            input_was_string = True

        all_embeddings = []
        mm_it_dataset = MMIT_Dataset(captions=captions, 
                                     image_ids=image_ids, 
                                     image_dir=self.image_dir,
                                     image_processor=self.image_processor
                                     )
        mm_it_collator = MMIT_Collator(self.tokenizer, caption_max_len=75)

        mm_it_dataloader = DataLoader(dataset=mm_it_dataset, 
                                      collate_fn=mm_it_collator, 
                                      num_workers=8, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,)

        for data in tqdm(mm_it_dataloader, desc="Inference Embeddings", disable=len(captions)<256):
            captions_inputs = data[0].to(self.device)
            
            images = data[1].to(self.device)
            if self.use_fp16 and images.dtype != torch.float16:
                images = images.half()

            text_embeddings = self.model.get_text_features(**captions_inputs)
            image_embeddings = self.model.get_image_features(images)
            
            embeddings = text_embeddings + image_embeddings
            
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings
    
    @torch.no_grad()
    def encode_image(self, image_ids: Union[List[str], str],  batch_size: int=256, max_length: int=77) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        all_embeddings = []
        image_dataset = Image_Dataset(image_ids=image_ids, 
                                     image_dir=self.image_dir,
                                     image_processor=self.image_processor
                                     )
        image_collator = Image_Collator(self.tokenizer, caption_max_len=312)

        image_dataloader = DataLoader(dataset=image_dataset, 
                                      collate_fn=image_collator, 
                                      num_workers=8, 
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,)

        for data in tqdm(image_dataloader, desc="Inference Image Embeddings"):
            
            images = data.to(self.device)
            if self.use_fp16 and images.dtype != torch.float16:
                images = images.half()
            


            embeddings = self.model.get_image_features(images)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        return all_embeddings