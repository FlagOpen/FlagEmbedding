import argparse
import os
import json
import random
import copy
import multiprocessing

from FlagEmbedding import FlagModel

from agent import GPTAgent, LLMAgent, LLMInstructAgent
from utils import generate_llm_dpo_train_data
from prompts import get_additional_info_generation_prompt


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
    parser.add_argument('--retrieval_model_name', type=str, default="bge-large-en-v1.5")
    parser.add_argument('--pooling_method', type=str, default='cls')
    parser.add_argument('--retrieval_query_prompt', type=str,
                        default="Represent this sentence for searching relevant passages: ")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset_path', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./synthetic")
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--dpo_num', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--normalize_embeddings', type=str, default='True')
    parser.add_argument('--use_rule1', type=str, default='True')

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
    retrieval_model_name = opt.retrieval_model_name
    retrieval_query_prompt = opt.retrieval_query_prompt
    dataset_path = opt.dataset_path
    output_dir = opt.output_dir
    max_length = opt.max_length
    batch_size = opt.batch_size
    threshold = opt.threshold
    dpo_num = opt.dpo_num
    pooling_method = opt.pooling_method
    dataset_name = opt.dataset_name
    normalize_embeddings = opt.normalize_embeddings
    if normalize_embeddings == 'False':
        normalize_embeddings = False
    else:
        normalize_embeddings = True
    use_rule1 = opt.use_rule1
    if use_rule1 == 'False':
        use_rule1 = False
    else:
        use_rule1 = True

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
        # continue
        if not os.path.isdir(os.path.join(dataset_path, file_path)):
            continue
        tmp_output_dir = os.path.join(output_dir, file_path)
        os.makedirs(tmp_output_dir, exist_ok=True)
        queries_output_dir = os.path.join(tmp_output_dir, 'queries.json')
        answers_output_dir = os.path.join(tmp_output_dir, 'answers.json')

        if file_path != 'cqadupstack':
            corpus_path = os.path.join(dataset_path, file_path, 'corpus.json')

            corpus = json.load(open(corpus_path))
        else:
            corpus = []
            for sub_file in os.listdir(os.path.join(dataset_path, file_path)):
                if not os.path.isdir(os.path.join(dataset_path, file_path, sub_file)):
                    continue
                corpus_path = os.path.join(dataset_path, file_path, sub_file, 'corpus.json')

                corpus.extend(json.load(open(corpus_path)))
                # old_corpus = copy.deepcopy(corpus)

        queries_corpus = json.load(open(queries_output_dir))

        queries_corpus_list = []
        if os.path.exists(answers_output_dir):
            queries_corpus_list = json.load(open(answers_output_dir))

        for idx in range(dpo_num - len(queries_corpus_list)):
            prompts = [get_additional_info_generation_prompt(file_path, qc['query']) for qc in queries_corpus]
            outputs = llm.generate(
                prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            queries_corpus_list.append(copy.deepcopy(queries_corpus))
            for i, output in enumerate(outputs):
                queries_corpus_list[-1][i]['answer'] = output
                queries_corpus_list[-1][i]['new_query'] = 'Generate the topic about this passage: ' + output

            with open(answers_output_dir, 'w') as f:
                json.dump(queries_corpus_list, f)

    retrieval_model = FlagModel(retrieval_model_name,
                                query_instruction_for_retrieval=retrieval_query_prompt,
                                pooling_method=pooling_method,
                                use_fp16=True,
                                trust_remote_code=True,
                                normalize_embeddings=normalize_embeddings)

    for file_path in os.listdir(dataset_path):
        if dataset_name is not None:
            if file_path != dataset_name:
                continue

        if not os.path.isdir(os.path.join(dataset_path, file_path)):
            continue
        tmp_output_dir = os.path.join(output_dir, file_path)
        os.makedirs(tmp_output_dir, exist_ok=True)
        answers_output_dir = os.path.join(tmp_output_dir, 'answers.json')
        llm_data_output_dir = os.path.join(tmp_output_dir, 'train.jsonl')

        try:
            queries_corpus_list = json.load(open(answers_output_dir))
            queries_corpus_list = queries_corpus_list[:dpo_num]
            for i in range(len(queries_corpus_list)):
                queries_corpus_list[i] = queries_corpus_list[i]
        except:
            continue

        tmp_corpus = copy.deepcopy([e['passage'] for e in queries_corpus_list[0]])

        print(len(tmp_corpus), file_path)

        llm_train_data = generate_llm_dpo_train_data(queries_corpus_list, 'answer', 'passage', retrieval_model,
                                                     threshold, batch_size, max_length, use_rule1)
        if model_type == 'llm_instruct':
            queries = [get_additional_info_generation_prompt(file_path, e['prompt']) for e in llm_train_data]

            for i in range(len(llm_train_data)):
                llm_train_data[i]['prompt'] = queries[i]
                llm_train_data[i]['chosen'] = llm_train_data[i]['chosen']
                llm_train_data[i]['rejected'] = llm_train_data[i]['rejected']
        elif model_type == 'llm':
            queries = [
                '###Instruction:\n' + get_additional_info_generation_prompt(file_path, e['prompt']) + '\n###Response:\n'
                for e in llm_train_data]
            for i in range(len(llm_train_data)):
                llm_train_data[i]['prompt'] = queries[i]
        else:
            queries = [get_additional_info_generation_prompt(file_path, e['prompt']) for e in llm_train_data]
            for i in range(len(llm_train_data)):
                llm_train_data[i]['prompt'] = queries[i]

        with open(llm_data_output_dir, 'w') as f:
            for d in llm_train_data:
                f.write(json.dumps(d) + '\n')


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    opt = parse_option()
    main(opt)