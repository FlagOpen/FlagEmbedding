import argparse
import json
import random
import numpy as np
import faiss
from tqdm import tqdm
import requests
from FlagEmbedding import FlagModel
from concurrent.futures import ThreadPoolExecutor

import os
import requests
import json
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--tei_url', type=str)
    parser.add_argument('--encode_batch_size', type=int,default=64)
    parser.add_argument('--thread_num', type=int,default=8,help="thread num only for tei speed up ")
    parser.add_argument('--input_file', default=None, type=str)
    parser.add_argument('--candidate_pool', default=None, type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--range_for_sampling', default="10-210", type=str, help="range to sample negatives")
    parser.add_argument('--use_gpu_for_searching', action='store_true', help='use faiss-gpu')
    parser.add_argument('--negative_number', default=15, type=int, help='the number of negatives')
    parser.add_argument('--query_instruction_for_retrieval', default="")

    return parser.parse_args()
def post_embed(tei_url, text_list, normalize=True, truncate=True, batch_size=64,thread_num=8):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    all_embeddings = []

    # 创建线程池
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        # 定义一个函数来处理每个批次
        def process_batch(batch_texts):
            data = {
                "inputs": batch_texts,
                "normalize": normalize,
                "truncate": truncate
            }
            # 发送请求
            response = requests.post(tei_url, headers=headers, data=json.dumps(data), timeout=60)
            # 检查响应状态
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code}, {response.text}")
            # 返回响应的嵌入结果
            return response.json()

        # 分批处理文本列表
        batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
        # 使用线程池执行请求，并保证顺序
        for embeddings in tqdm(executor.map(process_batch, batches), total=len(batches), desc="embedding"):
            all_embeddings.extend(embeddings)

    # 将所有嵌入结果转换为NumPy数组
    all_embeddings = np.array(all_embeddings)
    return all_embeddings

def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(index,
                 query,
                 topk: int = 200,
                 batch_size: int = 64):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool):
        line = json.loads(line.strip())
        corpus.append(line['text'])
    return corpus


def find_knn_neg(embed_fn,input_file, candidate_pool, output_file, sample_range, negative_number, use_gpu):
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file):
        line = json.loads(line.strip())
        if line['query'].strip()=="":
            continue
        train_data.append(line)
        corpus.extend([item.strip() for item in line['pos'] if item.strip()!="" ])
        if 'neg' in line:
            corpus.extend([item.strip() for item in line['neg'] if item.strip()!="" ])
        queries.append(line['query'])
        
    

    if candidate_pool is not None:
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus))

    print(f'inferencing embedding for corpus (number={len(corpus)})--------------')

    p_vecs = embed_fn(corpus)

    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = embed_fn(queries)

    print('create index and search------------------')
    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1])
    assert len(all_inxs) == len(train_data)

    for i, data in enumerate(train_data):
        query = data['query']
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1: break
            if corpus[inx] not in data['pos'] and corpus[inx] != query:
                filtered_inx.append(inx)

        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        data['neg'] = [corpus[inx] for inx in filtered_inx]
    directory = os.path.dirname(output_file)
    
    os.makedirs(directory, exist_ok=True)
    with open(output_file, 'w') as f:
        for data in train_data:
            if len(data['neg']) < negative_number:
                samples = random.sample(corpus, negative_number - len(data['neg']) + len(data['pos']))
                samples = [sent for sent in samples if sent not in data['pos']]
                data['neg'].extend(samples[: negative_number - len(data['neg'])])
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = get_args()
    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    if args.model_name_or_path:
        
        model = FlagModel(args.model_name_or_path, query_instruction_for_retrieval=args.query_instruction_for_retrieval)

        embedd_fn = lambda corpus :  model.encode(corpus, batch_size=args.encode_batch_size)
        
    elif args.tei_url: 

        embedd_fn = lambda corpus : post_embed(args.tei_url,corpus,batch_size=args.encode_batch_size)

    else:

        raise Exception("Both of model_name_or_path or tei_url is NULL !! ")
    
    find_knn_neg(embedd_fn,
        input_file=args.input_file,
        candidate_pool=args.candidate_pool,
        output_file=args.output_file,
        sample_range=sample_range,
        negative_number=args.negative_number,
        use_gpu=args.use_gpu_for_searching)
