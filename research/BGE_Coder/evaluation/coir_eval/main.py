import os
import json
import coir
from transformers import HfArgumentParser

from arguments import COIREvalArgs, COIREvalModelArgs
from prompts import get_task_def_by_task_name
from FlagEmbedding import FlagLLMModel, FlagModel


def get_model(model_args: COIREvalModelArgs):
    embedder_name_or_path = model_args.embedder_name_or_path

    if model_args.embedder_model_class == "encoder-only-base":
        embedder = FlagModel(
            model_name_or_path=embedder_name_or_path,
            normalize_embeddings=model_args.normalize_embeddings,
            pooling_method=model_args.pooling_method,
            use_fp16=model_args.use_fp16,
            query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
            query_instruction_format=model_args.query_instruction_format_for_retrieval,
            devices=model_args.devices,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
            batch_size=model_args.embedder_batch_size,
            query_max_length=model_args.embedder_query_max_length,
            passage_max_length=model_args.embedder_passage_max_length,
        )
    elif model_args.embedder_model_class == "decoder-only-base":
        embedder = FlagLLMModel(
            model_name_or_path=embedder_name_or_path,
            normalize_embeddings=model_args.normalize_embeddings,
            pooling_method=model_args.pooling_method,
            use_fp16=model_args.use_fp16,
            query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
            query_instruction_format=model_args.query_instruction_format_for_retrieval,
            devices=model_args.devices,
            examples_for_task=model_args.examples_for_task,
            examples_instruction_format=model_args.examples_instruction_format,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
            batch_size=model_args.embedder_batch_size,
            query_max_length=model_args.embedder_query_max_length,
            passage_max_length=model_args.embedder_passage_max_length,
        )
    else:
        raise ValueError(f"Invalid model class: {model_args.embedder_model_class}")
    embedder.model.config._name_or_path = model_args.embedder_name_or_path

    class CustomFlagModel:
        def __init__(self, model):
            self.model = model

        def encode_queries(self, queries, show_progress_bar, convert_to_tensor, **kwargs):
            if isinstance(queries, str):
                queries = [queries]

            if isinstance(queries[0], dict):
                queries = [(e.get('title') + ' ' + e['text']).strip() for e in queries]

            return self.model.encode_queries(queries, **kwargs)

        def encode_corpus(self, corpus, show_progress_bar, convert_to_tensor, **kwargs):
            if isinstance(corpus, str):
                corpus = [corpus]

            if isinstance(corpus[0], dict):
                corpus = [(e.get('title') + ' ' + e['text']).strip() for e in corpus]

            return self.model.encode_corpus(corpus, **kwargs)

        def encode(self, corpus, show_progress_bar, convert_to_tensor, **kwargs):
            if isinstance(corpus, str):
                corpus = [corpus]

            if isinstance(corpus[0], dict):
                corpus = [(e.get('title') + ' ' + e['text']).strip() for e in corpus]

            return self.model.encode(corpus, **kwargs)

    return CustomFlagModel(embedder)


def main(
    eval_args: COIREvalArgs,
    model_args: COIREvalModelArgs
):
    model = get_model(model_args)

    output_folder = os.path.join(eval_args.output_dir, os.path.basename(model.model.model.config._name_or_path))

    all_task = eval_args.tasks
    if not isinstance(all_task, list):
        all_task = [all_task]

    all_results = {}
    for task_name in all_task:
        save_path = os.path.join(output_folder, f"{task_name}.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                all_results[task_name] = results['metrics']
                continue

        tmp_task = coir.get_tasks(tasks=[task_name])
        evaluation = coir.COIR(tasks=tmp_task,
                               batch_size=model_args.embedder_batch_size)

        model.model.stop_self_pool()

        if eval_args.use_special_instructions:
            model.model.query_instruction_for_retrieval = get_task_def_by_task_name(task_name)

        results = evaluation.run(model, output_folder=output_folder)
        all_results[task_name] = results[task_name]

    csn_result = 0
    csn_num = 0
    csn_ccr_result = 0
    csn_ccr_num = 0
    pop_keys = []
    all_result = 0
    all_num = 0
    for k in all_results.keys():
        if 'CodeSearchNet-ccr' in k:
            csn_ccr_result += all_results[k]['NDCG']['NDCG@10']
            csn_ccr_num += 1
            pop_keys.append(k)
        elif 'CodeSearchNet' in k:
            csn_result += all_results[k]['NDCG']['NDCG@10']
            csn_num += 1
            pop_keys.append(k)
        else:
            all_result += all_results[k]['NDCG']['NDCG@10']
            all_num += 1
    if csn_num > 0:
        print('Using CodeSearchNet')
        all_result += csn_result / csn_num
        all_num += 1
    if csn_ccr_num > 0:
        print('Using CodeSearchNet-ccr')
        all_result += csn_ccr_result / csn_ccr_num
        all_num += 1
    new_results = {}
    for k in all_results:
        if k in pop_keys:
            continue
        new_results[k] = all_results[k]['NDCG']['NDCG@10']
    if csn_num > 0:
        new_results['CodeSearchNet'] = csn_result / csn_num
    if csn_ccr_num > 0:
        new_results['CodeSearchNet_CCR'] = csn_ccr_result / csn_ccr_num
    new_results['all'] = all_result / all_num

    print(new_results)

    with open(os.path.join(output_folder, 'OVERALL-results.json'), 'w') as f:
        json.dump(new_results, f)


if __name__ == "__main__":
    parser = HfArgumentParser((
        COIREvalArgs,
        COIREvalModelArgs
    ))
    eval_args, model_args = parser.parse_args_into_dataclasses()
    main(eval_args, model_args)
