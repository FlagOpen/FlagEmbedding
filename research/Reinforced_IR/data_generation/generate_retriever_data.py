import argparse
import os
import json
import multiprocessing

from agent import GPTAgent, LLMAgent, LLMInstructAgent
from prompts import get_additional_info_generation_prompt, TASK_DICT
from FlagEmbedding import FlagModel
from utils import generate_bge_train_data

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
    parser.add_argument('--filter_data', type=bool, default=False)
    parser.add_argument('--filter_num', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--emb_save_dir', type=str, default=None)
    parser.add_argument('--ignore_prefix', type=bool, default=False)
    parser.add_argument('--normalize_embeddings', type=str, default='True')
    parser.add_argument('--neg_type', type=str, default='95neg')

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
    pooling_method = opt.pooling_method
    retrieval_query_prompt = opt.retrieval_query_prompt
    max_length = opt.max_length
    batch_size = opt.batch_size
    dataset_path = opt.dataset_path
    output_dir = opt.output_dir
    filter_data = opt.filter_data
    filter_num = opt.filter_num
    dataset_name = opt.dataset_name
    emb_save_dir = opt.emb_save_dir
    ignore_prefix = opt.ignore_prefix
    normalize_embeddings = opt.normalize_embeddings
    if normalize_embeddings == 'False':
        normalize_embeddings = False
    else:
        normalize_embeddings = True
    neg_type = opt.neg_type

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
        os.makedirs(tmp_output_dir, exist_ok=True)
        queries_output_dir = os.path.join(tmp_output_dir, 'queries.json')
        answers_output_dir = os.path.join(tmp_output_dir, 'answers.json')

        queries_corpus = json.load(open(queries_output_dir))

        if os.path.exists(answers_output_dir):
            pass
        else:
            prompts = [get_additional_info_generation_prompt(file_path, qc['query']) for qc in queries_corpus]
            outputs = llm.generate(
                prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            for i in range(len(outputs)):
                queries_corpus[i]['answer'] = 'Generate the topic about this passage: ' + outputs[i]

            with open(answers_output_dir, 'w') as f:
                json.dump(queries_corpus, f)

    retrieval_model = FlagModel(retrieval_model_name,
                                query_instruction_for_retrieval=retrieval_query_prompt,
                                pooling_method=pooling_method,
                                use_fp16=True,
                                normalize_embeddings=normalize_embeddings)

    for file_path in os.listdir(dataset_path):
        if dataset_name is not None:
            if file_path != dataset_name:
                continue
        if not os.path.isdir(os.path.join(dataset_path, file_path)):
            continue
        tmp_output_dir = os.path.join(output_dir, file_path)
        answers_output_dir = os.path.join(tmp_output_dir, 'answers.json')
        retrieval_data_output_dir = os.path.join(tmp_output_dir, 'train.jsonl')

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

        old_corpus = corpus

        queries_corpus = json.load(open(answers_output_dir))

        corpus = [c['passage'] for c in queries_corpus]

        corpus.extend(old_corpus)

        print('corpus length:', len(corpus), ';', 'queries length:', len(queries_corpus))

        if emb_save_dir is not None:
            if file_path in ['cqadupstack', 'webis-touche2020']:
                emb_save_path = os.path.join(emb_save_dir, file_path, 'tmp_corpus.npy')
            else:
                emb_save_path = os.path.join(emb_save_dir, file_path, 'corpus.npy')
        else:
            emb_save_path = None

        bge_train_data = generate_bge_train_data(retrieval_model, batch_size, max_length,
                                                 queries_corpus, 'passage', corpus, filter_data, filter_num,
                                                 emb_save_path, ignore_prefix, neg_type)

        with open(retrieval_data_output_dir, 'w') as f:
            for d in bge_train_data:
                f.write(json.dumps(d) + '\n')


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    opt = parse_option()
    main(opt)