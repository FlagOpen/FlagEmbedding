import argparse
import os
import json
import random
import multiprocessing

from agent import GPTAgent, LLMAgent, LLMInstructAgent
from prompts import get_query_generation_prompt, get_quality_control_prompt

def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--generate_model_path', type=str, default="gpt-4o-mini")
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--model_type', type=str, default="llm")
    parser.add_argument('--train_num', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--dataset_path', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./synthetic")
    parser.add_argument('--dataset_name', type=str, default=None)

    opt = parser.parse_args()

    return opt


def main(opt):
    generate_model_path = opt.generate_model_path
    api_key = opt.api_key
    base_url = opt.base_url
    temperature = opt.temperature
    gpu_memory_utilization = opt.gpu_memory_utilization
    tensor_parallel_size = opt.tensor_parallel_size
    top_p = opt.top_p
    max_tokens = opt.max_tokens
    model_type = opt.model_type
    train_num = opt.train_num
    train_ratio = opt.train_ratio
    dataset_path = opt.dataset_path
    output_dir = opt.output_dir
    dataset_name = opt.dataset_name

    """
    dataset_path - data name - corpus.json
    output_dir - data name - queries.json
    """

    if model_type == 'llm':
        llm = LLMAgent(generate_model_path=generate_model_path,
                       gpu_memory_utilization=gpu_memory_utilization,
                       tensor_parallel_size=tensor_parallel_size)
    elif model_type == 'llm_instruct':
        llm = LLMInstructAgent(generate_model_path=generate_model_path,
                               gpu_memory_utilization=gpu_memory_utilization,
                               tensor_parallel_size=tensor_parallel_size)
    else:
        llm = GPTAgent(model_name=generate_model_path,
                       api_key=api_key,
                       base_url=base_url)

    for file_path in os.listdir(dataset_path):
        if dataset_name is not None:
            if file_path != dataset_name:
                continue
        if not os.path.isdir(os.path.join(dataset_path, file_path)):
            continue
        tmp_output_dir = os.path.join(output_dir, file_path)
        os.makedirs(tmp_output_dir, exist_ok=True)
        queries_output_dir = os.path.join(tmp_output_dir, 'queries.json')
        if file_path != 'cqadupstack':
            corpus_path = os.path.join(dataset_path, file_path, 'corpus.json')

            corpus = json.load(open(corpus_path))
        else:
            corpus = []
            for sub_file in os.listdir(os.path.join(dataset_path, file_path)):
                corpus_path = os.path.join(dataset_path, file_path, sub_file, 'corpus.json')

                corpus.extend(json.load(open(corpus_path)))
        random.shuffle(corpus)
        if train_ratio is not None:
            train_num = int(train_ratio * len(corpus))
        if train_num is not None:
            corpus = corpus[:train_num]

        ### generate queries for each corpus
        if not os.path.exists(queries_output_dir):
            prompts = [get_query_generation_prompt(file_path, c[:8000], use_examples=True) for c in corpus]
            generated_queries = llm.generate(
                prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            qualities_prompts = [get_quality_control_prompt(file_path, q, c) for (q, c) in
                                 zip(generated_queries, corpus)]

            generated_qualities = llm.generate(
                qualities_prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            print(generated_qualities)

            queries_corpus = []
            for i in range(len(generated_qualities)):
                if '1' in generated_qualities[i]:
                    queries_corpus.append(
                        {
                            'query': generated_queries[i],
                            'passage': corpus[i]
                        }
                    )

            with open(queries_output_dir, 'w') as f:
                json.dump(queries_corpus, f)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    opt = parse_option()
    main(opt)