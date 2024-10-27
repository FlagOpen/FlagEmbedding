"""
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--threads 16 \
--batch_size 32 \
--hits 1000 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False
"""
import os
import sys
import torch
import datasets
from tqdm import tqdm
from pprint import pprint
from dataclasses import dataclass, field
from transformers import HfArgumentParser, is_torch_npu_available
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from pyserini.output_writer import get_output_writer, OutputFormat


@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    query_instruction_for_retrieval: str = field(
        default=None,
        metadata={'help': 'query instruction for retrieval'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: en ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw', 
                  "nargs": "+"}
    )
    index_save_dir: str = field(
        default='./corpus-index',
        metadata={'help': 'Dir to index and docid. Corpus index path is `index_save_dir/{encoder_name}/index`. Corpus ids path is `index_save_dir/{encoder_name}/docid` .'}
    )
    result_save_dir: str = field(
        default='./search_results',
        metadata={'help': 'Dir to saving search results. Search results will be saved to `result_save_dir/{encoder_name}/{lang}.txt`'}
    )
    qa_data_dir: str = field(
        default='../qa_data',
        metadata={'help': 'Dir to qa data.'}
    )
    threads: int = field(
        default=1,
        metadata={'help': 'Maximum threads to use during search'}
    )
    batch_size: int = field(
        default=32,
        metadata={'help': 'Search batch size.'}
    )
    hits: int = field(
        default=1000,
        metadata={'help': 'Number of hits'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )


def get_query_encoder(model_args: ModelArgs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif is_torch_npu_available():
        device = torch.device("npu")
    else:
        device = torch.device("cpu")
    model = AutoQueryEncoder(
        encoder_dir=model_args.encoder,
        device=device,
        pooling=model_args.pooling_method,
        l2_norm=model_args.normalize_embeddings
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


def get_queries_and_qids(qa_data_dir: str, lang: str, add_instruction: bool=False, query_instruction_for_retrieval: str=None):
    topics_path = os.path.join(qa_data_dir, f"{lang}.jsonl")
    if not os.path.exists(topics_path):
        raise FileNotFoundError(f"{topics_path} not found")
    
    dataset = datasets.load_dataset('json', data_files=topics_path)['train']
    
    queries = []
    qids = []
    for data in dataset:
        qids.append(str(data['id']))
        queries.append(str(data['question']))
    if add_instruction and query_instruction_for_retrieval is not None:
        queries = [f"{query_instruction_for_retrieval}{query}" for query in queries]
    return queries, qids


def save_result(search_results, result_save_path: str, qids: list, max_hits: int):
    output_writer = get_output_writer(result_save_path, OutputFormat(OutputFormat.TREC.value), 'w',
                                      max_hits=max_hits, tag='Faiss', topics=qids,
                                      use_max_passage=False,
                                      max_passage_delimiter='#',
                                      max_passage_hits=1000)
    with output_writer:
        for topic, hits in search_results:
            output_writer.write(topic, hits)


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    
    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]
    
    query_encoder = get_query_encoder(model_args=model_args)
    
    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)
    
    index_save_dir = os.path.join(eval_args.index_save_dir, os.path.basename(encoder))
    if not os.path.exists(index_save_dir):
        raise FileNotFoundError(f"{index_save_dir} not found")
    searcher = FaissSearcher(
        index_dir=index_save_dir,
        query_encoder=query_encoder
    )
    
    print("==================================================")
    print("Start generating search results with model:", encoder)

    print('Generate search results of following languages: ', languages)
    for lang in languages:
        print("**************************************************")
        print(f"Start searching results of {lang} ...")
        
        result_save_path = os.path.join(eval_args.result_save_dir, os.path.basename(encoder), f"{lang}.txt")
        if not os.path.exists(os.path.dirname(result_save_path)):
            os.makedirs(os.path.dirname(result_save_path))
        
        if os.path.exists(result_save_path) and not eval_args.overwrite:
            print(f'Search results of {lang} already exists. Skip...')
            continue
        
        queries, qids = get_queries_and_qids(eval_args.qa_data_dir, lang=lang, add_instruction=model_args.add_instruction)
        
        search_results = []
        for start_idx in tqdm(range(0, len(queries), eval_args.batch_size), desc="Searching"):
            batch_queries = queries[start_idx : start_idx+eval_args.batch_size]
            batch_qids = qids[start_idx : start_idx+eval_args.batch_size]
            batch_search_results = searcher.batch_search(
                queries=batch_queries,
                q_ids=batch_qids,
                k=eval_args.hits,
                threads=eval_args.threads
            )
            search_results.extend([(_id, batch_search_results[_id]) for _id in batch_qids])
        
        save_result(
            search_results=search_results,
            result_save_path=result_save_path, 
            qids=qids, 
            max_hits=eval_args.hits
        )

    print("==================================================")
    print("Finish generating search results with following model:")
    pprint(model_args.encoder)


if __name__ == "__main__":
    main()
