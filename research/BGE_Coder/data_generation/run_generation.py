import os
import json
import time
import gc
import torch
import argparse
import random
from hashlib import md5
import multiprocessing as mp
from typing import List, Optional

from constant import TaskType, Language, CodeLanguage, NUM_HARD_NEGATIVES
from corpus_generator import CorpusGenerator
from triplet_generator import TripletGenerator
from search import get_top1


def compute_md5(text: str):
    return md5(text.encode()).hexdigest()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_type',
        type=str,
        required=True,
        help='The task type to generate data for',
        choices=[t.name for t in TaskType]
    )
    parser.add_argument(
        '--code_language',
        type=str,
        required=True,
        help='The code language to generate questions for.',
        choices=[c.name for c in CodeLanguage]
    )
    parser.add_argument(
        '--corpus_root',
        type=str,
        required=True,
        help='The root directory of the corpus data.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='The path to save the generated data'
    )
    parser.add_argument(
        '--examples_dir',
        type=str,
        default=None,
        help='The path to the examples directory. If not None, the examples will be used for few-shot generation.'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=3,
        help='The number of examples to use for few-shot generation. Default: 3'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='The cache directory'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='The language to generate for. ISO 639-1 code. Default: en',
        choices=[l.name for l in Language]
    )
    parser.add_argument(
        '--tgt_code_language',
        type=str,
        default=None,
        help='The target code language to generate code translations for.',
        choices=[c.name for c in CodeLanguage]
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='The number of examples to use for generation. Default: -1. Use all available examples.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen2.5-72B-Instruct',
        help='The model to use for generation. Default: Qwen2.5-72B-Instruct'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='open-source',
        help='The type of model to use for generation. Default: open-source',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='The port for vllm.'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=1,
        help='The number of processes to use for generation. Default: 1'
    )
    parser.add_argument(
        '--doc_length',
        type=str,
        default='len_0_500',
        help='The corpus length used to load dataset. Default: len_0_500'
    )
    parser.add_argument(
        '--external_path',
        type=str,
        default='',
        help='The corpus length used to load dataset. Default: len_0_500'
    )
    parser.add_argument(
        '--sim_model_name',
        type=str,
        default=None,
        help='The language of source corpus.'
    )
    parser.add_argument(
        '--max_corpus',
        type=int,
        default=500000,
        help='The max num of corpus to load.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite the existing data.'
    )
    parser.add_argument(
        '--debug_mode',
        action='store_true',
        help='Whether to open debug mode.'
    )
    parser.add_argument(
        '--gen_hard_neg',
        action='store_true',
        help='Whether to generate hard negatives.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for generating triplets using the same positive. Default: 42'
    )
    args = parser.parse_args()
    return args


def gen_triplets(
    model: str,
    model_type: str,
    port: int,
    positives: List[dict],
    task_type: str,
    language: str,
    code_language: str,
    tgt_code_language: str,
    examples_pool: Optional[List[dict]] = None,
    num_examples: int = 3,
    tqdm_desc: str = "Generating triplets",
    thread_count: int = 1,
    gen_cache_dir: Optional[str] = None,
    debug_mode: bool = False,
    gen_hard_neg: bool = False,
):
    triplet_generator = TripletGenerator(model, model_type, port, cache_dir=gen_cache_dir)
    triplets = triplet_generator.run(
        positives=positives,
        task_type=task_type,
        language=language,
        code_language=code_language,
        tgt_code_language=tgt_code_language,
        examples_pool=examples_pool,
        num_examples=num_examples,
        tqdm_desc=tqdm_desc,
        thread_count=thread_count,
        debug_mode=debug_mode,
        gen_hard_neg=gen_hard_neg,
        num_negatives=NUM_HARD_NEGATIVES,
    )
    return triplets


def get_save_path(
    save_dir: str,
    task_type: str,
    language: str,
    code_language: str,
    tgt_code_language: Optional[str] = None
):
    save_dir = os.path.join(save_dir, language, task_type)
    if tgt_code_language is not None:
        file_name = f"{language}-{code_language}-to-{tgt_code_language}-triplets.jsonl"
    else:
        file_name = f"{language}-{code_language}-triplets.jsonl"
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_path


def save_triplets(
    triplets: list,
    save_dir: str,
    task_type: str,
    language: str,
    code_language: str,
    tgt_code_language: Optional[str] = None
):
    if len(triplets) == 0:
        print(f"No triplets to save: {task_type} | {language} | {code_language} | {tgt_code_language}")
        return
    
    save_path = get_save_path(save_dir, task_type, language, code_language, tgt_code_language)
    query_md5s = set()
    pos_md5s = set()
    old_triplets = []
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                triplet = json.loads(line)
                old_triplets.append(triplet)
                query_md5s.add(compute_md5(triplet['query']))
                pos_md5s.add(compute_md5(triplet['pos'][0]))

    with open(save_path, 'w', encoding='utf-8') as f:
        for triplet in old_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
        
        for triplet in triplets:
            _query_md5 = compute_md5(triplet['query'])
            _pos_md5 = compute_md5(triplet['pos'][0])
            if _query_md5 in query_md5s or _pos_md5 in pos_md5s:
                continue
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
    print(f"Triplets saved to {save_path}")


def main(args):
    # set seed
    seed = args.seed
    if seed is not None:
        print(f"------------------- Seed set to {seed} -------------------")
        random.seed(seed)
    
    model = args.model
    model_type = args.model_type
    port = args.port

    num_samples = args.num_samples
    
    task_type = args.task_type
    language = args.language
    code_language = args.code_language
    tgt_code_language = args.tgt_code_language

    corpus_root = args.corpus_root
    corpus_dir = os.path.join(corpus_root, code_language)
    doc_length = args.doc_length.split()
    external_path = args.external_path.split()

    save_dir = args.save_dir
    cache_dir = args.cache_dir
    num_processes = min(args.num_processes, int(mp.cpu_count() * 0.8))
    overwrite = args.overwrite
    debug_mode = args.debug_mode
    gen_hard_neg = args.gen_hard_neg
    
    save_path = get_save_path(save_dir, task_type, language, code_language, tgt_code_language)
    # if os.path.exists(save_path) and not overwrite:
        # data = []
        # with open(save_path) as f:
        #     for line in f:
        #         data.append(json.loads(line))
        # if len(data) >= num_samples * 0.8:
        #     print(f"Triplets already exist at {save_path}. Skipping generation.")
        #     return
        # else:
        #     print(f"Triplets already exist at {save_path}. But samples is really small, continue generation.")
        #     num_samples = int((num_samples - len(data)) * 1.25)  # consider the filtered samples

    corpus_generator = CorpusGenerator(cache_dir)

    examples_dir = args.examples_dir
    num_examples = args.num_examples
    if examples_dir is not None:
        # if task_type in ["single_turn_code_qa", "multi_turn_code_qa"]:
        #     examples_path = os.path.join(examples_dir, language, task_type, "sample_examples.json")
        if task_type in ["code_translation_retrieval"]:
            examples_path = os.path.join(examples_dir, language, task_type,
                                         f"{code_language}-to-{tgt_code_language}_sample_examples.json")
        else:
            examples_path = os.path.join(examples_dir, language, task_type, f"{code_language}_sample_examples.json")
        try:
            with open(examples_path, 'r', encoding='utf-8') as f:
                examples_pool = json.load(f)
                examples_pool = random.sample(examples_pool,
                                              min(30, len(examples_pool)))   # sample 30 examples for few-shot generation
        except:
            print(f'Error for loading examples from {examples_path}')
            examples_pool = None
    else:
        examples_pool = None

    positives, large_positives = corpus_generator.run(
        num_samples=num_samples,
        max_corpus=args.max_corpus,
        corpus_dir=corpus_dir,
        doc_length=doc_length,
        external_path=external_path,
        source_language=code_language
    )

    if task_type in ["code_modification_retrieval", "code_comparison_retrieval"]:
        top1_docs = get_top1([e['text'] for e in positives], args.sim_model_name, [e['text'] for e in large_positives])
        for i in range(len(top1_docs)):
            positives[i]['similar'] = top1_docs[i]
        gc.collect()
        torch.cuda.empty_cache()

    print("=================== Generate training data ===================")
    print(f'Task Type: {task_type} | Language: {language} | Code Language: {code_language} | Target Code Language: {tgt_code_language}')
    start_time = time.time()
    triplets = gen_triplets(
        model=model,
        model_type=model_type,
        port=port,
        positives=positives,
        task_type=task_type,
        language=language,
        code_language=code_language,
        tgt_code_language=tgt_code_language,
        examples_pool=examples_pool,
        num_examples=num_examples,
        thread_count=num_processes,
        gen_cache_dir=os.path.join(save_dir, language, task_type, "gen_cache_dir"),
        debug_mode=debug_mode,
        gen_hard_neg=gen_hard_neg,
    )
    save_triplets(
        triplets=triplets,
        save_dir=save_dir,
        task_type=task_type,
        language=language,
        code_language=code_language,
        tgt_code_language=tgt_code_language
    )
    end_time = time.time()
    print("=============================================================")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("=============================================================")
    print("DONE!")


if __name__ == "__main__":
    args = get_args()
    main(args)
