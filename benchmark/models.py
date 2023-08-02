import numpy as np
from typing import cast, List, Dict
import torch
from tqdm import tqdm
from mteb import DRESModel


class UniversalModel(DRESModel):
    def __init__(
            self,
            model_name_or_path: str = None,
            normlized: bool = False,
            num_gpus: int = None,
            query_instruction_for_retrieval: str = None,
            **kwargs
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        self.num_gpus = num_gpus
        if num_gpus == 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model = self.model.to(self.device)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normlized = normlized

        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)


    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        encode queries for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts)


    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        '''
        encode corpus for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        return self.encode(input_texts)


    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 256, **kwargs) -> np.ndarray:

        batch_size = min(batch_size, 256) * self.num_gpus
        self.model.eval()

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = last_hidden_state[:, 0]
            if self.normlized:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)







