import torch
import datasets
import numpy as np
from tqdm import tqdm
from mteb import DRESModel
from functools import partial
from torch.utils.data import DataLoader
from typing import cast, List, Dict, Union
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available
from transformers import PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding


def _transform_func(examples: Dict[str, List], 
                    tokenizer: PreTrainedTokenizerFast,
                    max_length: int) -> BatchEncoding:
    return tokenizer(examples['text'],
                     max_length=max_length,
                     padding=True,
                     return_token_type_ids=False,
                     truncation=True,
                     return_tensors='pt')


def _transform_func_v2(examples: Dict[str, List],
                    tokenizer: PreTrainedTokenizerFast,
                    max_length: int=8192,
                    ) -> BatchEncoding:
    
    inputs = tokenizer(examples['text'],
                     max_length=max_length - 1,
                     padding=False,
                     return_attention_mask=False,
                     truncation=True)
    inputs['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in inputs['input_ids']]
    inputs = tokenizer.pad(inputs, padding=True, return_attention_mask=True, return_tensors='pt')
    return inputs


class FlagDRESModel(DRESModel):
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            use_fp16: bool = True,
            query_instruction_for_retrieval: str = None,
            passage_instruction_for_retrieval: str = None,
            max_query_length: int = 512,
            max_passage_length: int = 8192,
            batch_size: int = 256,
            corpus_batch_size: int = 0,
            **kwargs
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if 'jina' in model_name_or_path:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.passage_instruction_for_retrieval = passage_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.corpus_batch_size = corpus_batch_size if corpus_batch_size > 0 else batch_size
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        
        if use_fp16: self.model.half()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if isinstance(queries[0], dict):
            if self.query_instruction_for_retrieval is not None:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q['text']) for q in queries]
            else:
                input_texts = [q['text'] for q in queries]
        else:
            if self.query_instruction_for_retrieval is not None:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
            else:
                input_texts = queries
        return self.encode(input_texts, max_length=self.max_query_length, batch_size=self.batch_size)

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        if isinstance(corpus[0], dict):
            if self.passage_instruction_for_retrieval is not None:
                input_texts = ['{}{} {}'.format(self.passage_instruction_for_retrieval, doc.get('title', ''), doc['text']).strip() for doc in corpus]
            else:
                input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            if self.passage_instruction_for_retrieval is not None:
                input_texts = self.passage_instruction_for_retrieval + corpus
            else:
                input_texts = corpus
        return self.encode(input_texts, max_length=self.max_passage_length, batch_size=self.corpus_batch_size)

    @torch.no_grad()
    def encode(self, sentences: List[str], max_length: int, batch_size: int, **kwargs) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
    
        dataset = datasets.Dataset.from_dict({'text': sentences})
        if self.pooling_method == 'last':
            dataset.set_transform(partial(_transform_func_v2, tokenizer=self.tokenizer, max_length=max_length))
        else:
            dataset.set_transform(partial(_transform_func, tokenizer=self.tokenizer, max_length=max_length))

        data_collator = DataCollatorWithPadding(self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            # pin_memory=True
            )
        
        all_embeddings = []
        for batch_data in tqdm(data_loader, desc='encoding', mininterval=10):
            batch_data = batch_data.to(self.device)
            # print(batch_data)
            last_hidden_state = self.model(**batch_data, return_dict=True).last_hidden_state
            # print(last_hidden_state)
            embeddings = self.pooling(last_hidden_state, batch_data['attention_mask']).float()
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        else:
            return all_embeddings
    
    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.pooling_method == 'last':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
