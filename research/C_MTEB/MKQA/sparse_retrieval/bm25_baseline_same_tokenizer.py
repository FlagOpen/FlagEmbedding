"""
# 1. Output Search Results with BM25
python bm25_baseline_same_tokenizer.py

# 2. Print and Save Evaluation Results
python step2-eval_sparse_mkqa.py \
--encoder bm25_same_tokenizer \
--languages ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32
"""
import os
import sys
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append("..")

from utils.normalize_text import normalize


tokenizer = AutoTokenizer.from_pretrained(
    'BAAI/bge-m3',
    use_fast=False,
)


def _map_func_corpus(examples):
    results = {}
    results['id'] = examples['id']
    results['contents'] = []
    
    inputs = tokenizer(
        examples['contents'],
        padding=False,
        truncation=True,
        max_length=512
    )
    input_ids_list = inputs['input_ids']
    for i in range(len(examples['id'])):
        token_ids = input_ids_list[i][1:-1]
        token_ids = [str(_id) for _id in token_ids]
        results['contents'].append(" ".join(token_ids))
    return results


def _map_func_query(examples):
    results = {}
    results['id'] = examples['id']
    results['question'] = []
    
    inputs = tokenizer(
        examples['question'],
        padding=False,
        truncation=True,
        max_length=512
    )
    input_ids_list = inputs['input_ids']
    for i in range(len(examples['id'])):
        token_ids = input_ids_list[i][1:-1]
        token_ids = [str(_id) for _id in token_ids]
        results['question'].append(" ".join(token_ids))
    return results


def generate_corpus(corpus_save_path: str):
    if os.path.exists(corpus_save_path):
        print("Corpus already exists. Skip generating ...")
        return

    corpus = datasets.load_dataset('BeIR/nq', 'corpus')['corpus']
    corpus_list = []
    for data in tqdm(corpus, desc="Generating corpus"):
        _id = str(data['_id'])
        content = f"{data['title']}\n{data['text']}".lower()
        content = normalize(content)
        corpus_list.append({"id": _id, "contents": content})
    
    corpus = datasets.Dataset.from_list(corpus_list)
    
    corpus = corpus.map(_map_func_corpus, batched=True, num_proc=48)
    
    corpus.to_json(corpus_save_path, force_ascii=False)


def generate_queries(qa_data_dir: str, lang: str, queries_save_dir: str):
    queries_save_path = os.path.join(queries_save_dir, f"{lang}.tsv")
    if os.path.exists(queries_save_path) and os.path.getsize(queries_save_path) > 0:
        return
    
    queries_path = os.path.join(qa_data_dir, f"{lang}.jsonl")
    queries = datasets.load_dataset('json', data_files=queries_path)['train']
    
    queries = queries.map(_map_func_query, batched=True, num_proc=48)
    
    queries_list = []
    for data in queries:
        _id = str(data['id'])
        query = data['question']
        queries_list.append({
            'id': _id,
            'content': query
        })
    
    with open(queries_save_path, 'w', encoding='utf-8') as f:
        for query in queries_list:
            line = f"{query['id']}\t{query['content']}"
            f.write(line + '\n')


def index(corpus_save_dir: str, index_save_dir: str):
    cmd = f"python3 -m pyserini.index.lucene \
            --collection JsonCollection \
            --input {corpus_save_dir} \
            --index {index_save_dir} \
            --generator DefaultLuceneDocumentGenerator \
            --threads 1 \
            --storePositions --storeDocvectors --storeRaw \
        "
    os.system(cmd)


def search(index_save_dir: str, queries_save_dir: str, lang: str, result_save_path: str):
    queries_save_path = os.path.join(queries_save_dir, f"{lang}.tsv")
    cmd = f"python3 -m pyserini.search.lucene \
            --index {index_save_dir} \
            --topics {queries_save_path} \
            --output {result_save_path} \
            --bm25 \
            --hits 1000 \
            --batch-size 128 \
            --threads 16 \
        "
    os.system(cmd)


def main():
    qa_data_dir = '../qa_data'
    bm25_dir = './bm25_baseline_same_tokenizer'
    
    result_save_dir = os.path.join('./search_results', 'bm25_same_tokenizer')
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    
    corpus_save_dir = os.path.join(bm25_dir, 'corpus')
    if not os.path.exists(corpus_save_dir):
        os.makedirs(corpus_save_dir)
    
    corpus_save_path = os.path.join(corpus_save_dir, 'corpus.jsonl')
    generate_corpus(corpus_save_path)
    
    index_save_dir = os.path.join(bm25_dir, 'index')
    if not os.path.exists(index_save_dir):
        os.makedirs(index_save_dir)
    index(corpus_save_dir, index_save_dir)
    
    queries_save_dir = os.path.join(bm25_dir, 'queries')
    if not os.path.exists(queries_save_dir):
        os.makedirs(queries_save_dir)
    
    languages = ['ar', 'fi', 'ja', 'ko', 'ru', 'es', 'sv', 'he', 'th', 'da', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'hu', 'vi', 'ms', 'km', 'no', 'tr', 'zh_cn', 'zh_hk', 'zh_tw']
    
    for lang in languages:
        generate_queries(qa_data_dir, lang, queries_save_dir)
        
        result_save_path = os.path.join(result_save_dir, f'{lang}.txt')
        search(index_save_dir, queries_save_dir, lang, result_save_path)


if __name__ == '__main__':
    main()
