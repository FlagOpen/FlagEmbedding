"""
# 1. Output Search Results with BM25
python bm25_baseline_same_tokenizer.py

# 2. Print and Save Evaluation Results
python step2-eval_sparse_mldr.py \
--encoder bm25_same_tokenizer \
--languages ar de es fr hi it ja ko pt ru th en zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10
"""
import os
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    'BAAI/bge-m3',
    use_fast=False,
)


def _map_func_corpus(examples):
    results = {}
    results['docid'] = examples['docid']
    results['text'] = []
    
    inputs = tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=8192
    )
    input_ids_list = inputs['input_ids']
    
    for i in range(len(examples['docid'])):
        token_ids = input_ids_list[i][1:-1]
        token_ids = [str(_id) for _id in token_ids]
        results['text'].append(" ".join(token_ids))
    return results


def _map_func_query(examples):
    results = {}
    results['query_id'] = examples['query_id']
    results['query'] = []
    
    inputs = tokenizer(
        examples['query'],
        padding=False,
        truncation=True,
        max_length=512
    )
    
    input_ids_list = inputs['input_ids']
    
    for i in range(len(examples['query_id'])):
        token_ids = input_ids_list[i][1:-1]
        token_ids = [str(_id) for _id in token_ids]
        results['query'].append(" ".join(token_ids))
    return results


def generate_corpus(lang: str, corpus_save_dir: str):
    corpus_save_path = os.path.join(corpus_save_dir, 'corpus.jsonl')
    if os.path.exists(corpus_save_path):
        return
    
    corpus = datasets.load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus')
    
    corpus = corpus.map(_map_func_corpus, batched=True, num_proc=48)
    
    corpus_list = [{'id': e['docid'], 'contents': e['text']} for e in tqdm(corpus, desc="Generating corpus")]
    corpus = datasets.Dataset.from_list(corpus_list)
    
    corpus.to_json(corpus_save_path, force_ascii=False)


def generate_queries(lang: str, queries_save_dir: str, split: str='test'):
    queries_save_path = os.path.join(queries_save_dir, f"{lang}.tsv")
    if os.path.exists(queries_save_path):
        return
    
    dataset = datasets.load_dataset('Shitao/MLDR', lang, split=split)
    
    dataset = dataset.map(_map_func_query, batched=True, num_proc=48)
    
    queries_list = []
    for data in dataset:
        queries_list.append({
            'id': data['query_id'],
            'content': data['query']
        })
    with open(queries_save_path, 'w', encoding='utf-8') as f:
        for query in queries_list:
            assert '\n' not in query['content'] and '\t' not in query['content']
            line = f"{query['id']}\t{query['content']}"
            f.write(line + '\n')


def index(corpus_save_dir: str, index_save_dir: str):
    cmd = f"python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input {corpus_save_dir} \
            --index {index_save_dir} \
            --generator DefaultLuceneDocumentGenerator \
            --threads 1 --optimize \
        "
    os.system(cmd)


def search(index_save_dir: str, queries_save_dir: str, lang: str, result_save_path: str):
    queries_save_path = os.path.join(queries_save_dir, f"{lang}.tsv")
    cmd = f"python -m pyserini.search.lucene \
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
    bm25_dir = './bm25_baseline_same_tokenizer'
    
    result_save_dir = os.path.join('./search_results', 'bm25_same_tokenizer')
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    
    for lang in ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']:
        save_dir = os.path.join(bm25_dir, lang)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        corpus_save_dir = os.path.join(save_dir, 'corpus')
        if not os.path.exists(corpus_save_dir):
            os.makedirs(corpus_save_dir)
        generate_corpus(lang, corpus_save_dir)
        
        index_save_dir = os.path.join(save_dir, 'index')
        if not os.path.exists(index_save_dir):
            os.makedirs(index_save_dir)
        index(corpus_save_dir, index_save_dir)
        
        generate_queries(lang, save_dir, split='test')
        
        result_save_path = os.path.join(result_save_dir, f'{lang}.txt')
        search(index_save_dir, save_dir, lang, result_save_path)


if __name__ == '__main__':
    main()
