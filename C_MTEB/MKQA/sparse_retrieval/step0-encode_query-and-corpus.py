"""
python step0-encode_query-and-corpus.py \
--encoder BAAI/bge-m3 \
--languages ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw \
--qa_data_dir ../qa_data \
--save_dir ./encoded_query-and-corpus \
--max_query_length 512 \
--max_passage_length 512 \
--batch_size 1024 \
--pooling_method cls \
--normalize_embeddings True
"""

import os
import sys
import json
import datasets
import numpy as np
from tqdm import tqdm
from pprint import pprint
from FlagEmbedding import BGEM3FlagModel
from dataclasses import dataclass, field
from transformers import HfArgumentParser

sys.path.append("..")

from utils.normalize_text import normalize


@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 in inference?'}
    )


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: en ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw', 
                  "nargs": "+"}
    )
    qa_data_dir: str = field(
        default='../qa_data',
        metadata={'help': 'Dir to qa data.'}
    )
    save_dir: str = field(
        default='./encoded_query-and-corpus',
        metadata={'help': 'Dir to save encoded query and corpus. Encoded query and corpus will be saved to `save_dir/{encoder_name}/{lang}/query_embd.tsv` and `save_dir/{encoder_name}/corpus/corpus_embd.jsonl`, individually.'}
    )
    max_query_length: int = field(
        default=512,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=512,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )

def get_model(model_args: ModelArgs):
    model = BGEM3FlagModel(
        model_name_or_path=model_args.encoder,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.fp16
    )
    return model


def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['en', 'ar', 'fi', 'ja', 'ko', 'ru', 'es', 'sv', 'he', 'th', 'da', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'hu', 'vi', 'ms', 'km', 'no', 'tr', 'zh_cn', 'zh_hk', 'zh_tw']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages


def parse_corpus(corpus: datasets.Dataset):
    corpus_list = []
    for data in tqdm(corpus, desc="Generating corpus"):
        _id = str(data['_id'])
        content = f"{data['title']}\n{data['text']}".lower()
        content = normalize(content)
        corpus_list.append({"id": _id, "content": content})
    
    corpus = datasets.Dataset.from_list(corpus_list)
    return corpus


def get_queries(qa_data_dir: str, lang: str):
    topics_path = os.path.join(qa_data_dir, f"{lang}.jsonl")
    if not os.path.exists(topics_path):
        raise FileNotFoundError(f"{topics_path} not found")
    
    dataset = datasets.load_dataset('json', data_files=topics_path)['train']
    
    queries_list = []
    for data in dataset:
        _id = str(data['id'])
        query = data['question']
        queries_list.append({
            'id': _id,
            'content': query
        })
    
    queries = datasets.Dataset.from_list(queries_list)
    return queries


def encode_and_save_corpus(corpus_save_path: str, model: BGEM3FlagModel, corpus: datasets.Dataset, max_passage_length: int=512, batch_size: int=256):
    docids = list(corpus["id"])
    vectors = model.encode(
        corpus["content"], 
        batch_size=batch_size, 
        max_length=max_passage_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False
    )['lexical_weights']
    
    encoded_corpus_list = []
    for docid, vector in zip(docids, vectors):
        for key, value in vector.items():
            vector[key] = int(np.ceil(value * 100))
        
        encoded_corpus_list.append({
            'id': docid,
            'contents': '',
            'vector': vector
        })
    
    with open(corpus_save_path, 'w', encoding='utf-8') as f:
        for line in tqdm(encoded_corpus_list, desc="Saving encoded corpus"):
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def encode_and_save_queries(queries_save_path: str, model: BGEM3FlagModel, queries: datasets.Dataset, max_query_length: int=512, batch_size: int=256):
    qids = list(queries["id"])
    vectors = model.encode(
        queries["content"], 
        batch_size=batch_size, 
        max_length=max_query_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False
    )['lexical_weights']
    
    encoded_queries_list = []
    for qid, vector in zip(qids, vectors):
        for key, value in vector.items():
            vector[key] = int(np.ceil(value * 100))
        
        topic_str = []
        for token in vector:
            topic_str += [str(token)] * vector[token]
        if len(topic_str) == 0:
            topic_str = "0"
        else:
            topic_str = " ".join(topic_str)
        encoded_queries_list.append(f"{str(qid)}\t{topic_str}")
    
    with open(queries_save_path, 'w', encoding='utf-8') as f:
        for line in tqdm(encoded_queries_list, desc="Saving encoded queries"):
            f.write(line + '\n')


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    # languages.reverse()
    
    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]
    
    model = get_model(model_args=model_args)
    
    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)
    
    print("==================================================")
    print("Start generating embedding with model:")
    print(model_args.encoder)
    
    print('Generating corpus embedding ...')
    
    corpus_save_dir = os.path.join(eval_args.save_dir, os.path.basename(encoder), 'corpus')
    if not os.path.exists(corpus_save_dir):
        os.makedirs(corpus_save_dir)
    corpus_save_path = os.path.join(corpus_save_dir, 'corpus_embd.jsonl')
    if os.path.exists(corpus_save_path) and os.path.getsize(corpus_save_path) > 0 and not eval_args.overwrite:
        print(f'Corpus embedding already exists. Skip...')
    else:
        corpus = datasets.load_dataset("BeIR/nq", 'corpus')['corpus']
        corpus = parse_corpus(corpus=corpus)
        encode_and_save_corpus(
            corpus_save_path=corpus_save_path,
            model=model,
            corpus=corpus,
            max_passage_length=eval_args.max_passage_length,
            batch_size=eval_args.batch_size
        )
    print('Generate query embedding of following languages: ', languages)
    for lang in languages:
        print("**************************************************")
        save_dir = os.path.join(eval_args.save_dir, os.path.basename(encoder), lang)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        queries_save_path = os.path.join(save_dir, 'query_embd.tsv')
        if os.path.exists(queries_save_path) and not eval_args.overwrite:
            print(f'Query embedding of {lang} already exists. Skip...')
            continue
        
        print(f"Start generating query embedding of {lang} ...")
        queries = get_queries(eval_args.qa_data_dir, lang)
        encode_and_save_queries(
            queries_save_path=queries_save_path,
            model=model,
            queries=queries,
            max_query_length=eval_args.max_query_length,
            batch_size=eval_args.batch_size
        )

    print("==================================================")
    print("Finish generating embeddings with following model:")
    pprint(model_args.encoder)


if __name__ == "__main__":
    main()
