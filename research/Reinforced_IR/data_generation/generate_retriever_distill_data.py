import argparse
import os
import json
import multiprocessing

from transformers import AutoTokenizer
from tqdm import tqdm

from prompts import rank_prompt
from agent import GPTAgent, LLMAgent, LLMInstructAgent
from utils import generate_bge_train_data, get_distill_data

def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--generate_model_path', type=str, default="Meta-Llama-3-8B")
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)

    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--model_type', type=str, default="llm")

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

    dataset_path = opt.dataset_path
    output_dir = opt.output_dir
    dataset_name = opt.dataset_name

    """
    dataset_path - data name - corpus.json
    output_dir - data name - queries.json / answers.json / train.jsonl
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
        retrieval_data_output_dir = os.path.join(tmp_output_dir, 'train.jsonl')
        bge_train_data = []
        with open(retrieval_data_output_dir, 'r') as f:
            for line in f:
                bge_train_data.append(json.loads(line))

        if model_type != 'gpt':
            tmp_tokenizer = AutoTokenizer.from_pretrained(generate_model_path)
        else:
            tmp_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
        prompts = []
        for d in tqdm(bge_train_data, desc='generate train data'):
            passages = []
            passages.extend(d['pos'])
            passages.extend(d['neg'])
            passages_ids = tmp_tokenizer(passages, max_length=512, truncation=True)['input_ids']
            passages = tmp_tokenizer.batch_decode(passages_ids)
            prompts.append(
                rank_prompt.format(
                    num=len(passages),
                    query=d['query'],
                    passages='\n'.join([f'[{i}] {passages[i]}' for i in range(len(passages))])
                )
            )

        bge_train_data = get_distill_data(
            llm_for_rank=llm,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            train_data=bge_train_data,
            prompts=prompts,
        )

        with open(retrieval_data_output_dir, 'w') as f:
            for d in bge_train_data:
                f.write(json.dumps(d) + '\n')


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    opt = parse_option()
    main(opt)