"""
# 1. Output Search Results with BM25
python bm25_baseline.py

# 2. Print and Save Evaluation Results
python step2-eval_sparse_mkqa.py \
--encoder bm25 \
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

sys.path.append("..")

from utils.normalize_text import normalize


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
    corpus.to_json(corpus_save_path, force_ascii=False)


def generate_queries(qa_data_dir: str, lang: str, queries_save_dir: str):
    queries_save_path = os.path.join(queries_save_dir, f"{lang}.tsv")
    if os.path.exists(queries_save_path) and os.path.getsize(queries_save_path) > 0:
        return
    
    queries_path = os.path.join(qa_data_dir, f"{lang}.jsonl")
    queries = datasets.load_dataset('json', data_files=queries_path)['train']
    
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
    cmd = f"python -m pyserini.index.lucene \
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
    # Note: Use `--lang {lang}` will cause the performance degradation, since the query and corpus are in different languages.
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
    bm25_dir = './bm25_baseline'
    
    qa_data_dir = '../qa_data'
    
    result_save_dir = os.path.join('./search_results', 'bm25')
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
