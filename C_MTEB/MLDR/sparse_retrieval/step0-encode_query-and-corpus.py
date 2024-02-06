"""
python step0-encode_query-and-corpus.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--save_dir ./encoded_query-and-corpus \
--max_query_length 512 \
--max_passage_length 8192 \
--batch_size 1024 \
--corpus_batch_size 4 \
--pooling_method cls \
--normalize_embeddings True
"""

import os
import json
import datasets
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from dataclasses import dataclass, field
from transformers import HfArgumentParser


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
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh', 
                  "nargs": "+"}
    )
    save_dir: str = field(
        default='./encoded_query-and-corpus',
        metadata={'help': 'Dir to save encoded query and corpus. Encoded query and corpus will be saved to `save_dir/{encoder_name}/{lang}/query_embd.tsv` and `save_dir/{encoder_name}/{lang}/corpus/corpus_embd.jsonl`, individually.'}
    )
    max_query_length: int = field(
        default=512,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=8192,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    corpus_batch_size: int = field(
        default=4,
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
    avaliable_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages


def load_corpus(lang: str):
    corpus = datasets.load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus')
    
    corpus_list = [{'id': e['docid'], 'content': e['text']} for e in tqdm(corpus, desc="Generating corpus")]
    corpus = datasets.Dataset.from_list(corpus_list)
    return corpus


def get_queries(lang: str, split: str='test'):
    dataset = datasets.load_dataset('Shitao/MLDR', lang, split=split)
    
    queries_list = []
    for data in dataset:
        queries_list.append({
            'id': data['query_id'],
            'content': data['query']
        })
    
    queries = datasets.Dataset.from_list(queries_list)
    return queries


def encode_corpus(model: BGEM3FlagModel, corpus: datasets.Dataset, max_passage_length: int=8192, corpus_batch_size: int=4):
    docids = list(corpus["id"])
    vectors = model.encode(
        corpus["content"], 
        batch_size=corpus_batch_size, 
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
    return encoded_corpus_list


def encode_queries(model: BGEM3FlagModel, queries: datasets.Dataset, max_query_length: int=512, batch_size: int=256):
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
        encoded_queries_list.append(f"{str(qid)}\t{topic_str}\n")
    return encoded_queries_list


def save_result(encoded_queries_list: list, encoded_corpus_list: list, save_dir: str):
    queries_save_path = os.path.join(save_dir, 'query_embd.tsv')
    corpus_save_path = os.path.join(save_dir, 'corpus', 'corpus_embd.jsonl')
    if not os.path.exists(os.path.dirname(corpus_save_path)):
        os.makedirs(os.path.dirname(corpus_save_path))
    
    with open(queries_save_path, 'w', encoding='utf-8') as f:
        for line in tqdm(encoded_queries_list, desc="Saving encoded queries"):
            f.write(line)
    
    with open(corpus_save_path, 'w', encoding='utf-8') as f:
        for line in tqdm(encoded_corpus_list, desc="Saving encoded corpus"):
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


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

    print('Generate embedding of following languages: ', languages)
    for lang in languages:
        print("**************************************************")
        save_dir = os.path.join(eval_args.save_dir, os.path.basename(encoder), lang)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.exists(os.path.join(save_dir, 'corpus', 'corpus_embd.jsonl')) and not eval_args.overwrite:
            print(f'Embedding of {lang} already exists. Skip...')
            continue
        
        print(f"Start generating query and corpus embedding of {lang} ...")
        queries = get_queries(lang, split='test')
        encoded_queries_list = encode_queries(
            model=model,
            queries=queries,
            max_query_length=eval_args.max_query_length,
            batch_size=eval_args.batch_size
        )
        
        corpus = load_corpus(lang)
        encoded_corpus_list = encode_corpus(
            model=model,
            corpus=corpus,
            max_passage_length=eval_args.max_passage_length,
            corpus_batch_size=eval_args.corpus_batch_size
        )
        
        save_result(
            encoded_queries_list=encoded_queries_list,
            encoded_corpus_list=encoded_corpus_list,
            save_dir=save_dir
        )
    
    print("==================================================")
    print("Finish generating embeddings with model:")
    print(model_args.encoder)


if __name__ == "__main__":
    main()
