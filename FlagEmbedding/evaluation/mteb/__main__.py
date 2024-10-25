import mteb


from transformers import HfArgumentParser

from FlagEmbedding import FlagAutoModel, FlagAutoReranker
from FlagEmbedding.abc.evaluation import AbsModelArgs, AbsEmbedder, AbsReranker, AbsEvaluator


from utils.arguments import MTEBEvalArgs
from utils.prompts import get_task_def_by_task_name_and_type, tasks_desc


def get_models(model_args: AbsModelArgs):
    retriever = FlagAutoModel.from_finetuned(
        model_name_or_path=model_args.embedder_name_or_path,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.use_fp16,
        query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
        query_instruction_format=model_args.query_instruction_format_for_retrieval,
        devices=model_args.devices,
        examples_for_task=model_args.examples_for_task,
        examples_instruction_format=model_args.examples_instruction_format,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir
    )
    reranker = None
    if model_args.reranker_name_or_path is not None:
        reranker = FlagAutoReranker.from_finetuned(
            model_name_or_path=model_args.reranker_name_or_path,
            peft_path=model_args.reranker_peft_path,
            use_fp16=model_args.use_fp16,
            use_bf16=model_args.use_bf16,
            query_instruction_for_rerank=model_args.query_instruction_for_rerank,
            query_instruction_format=model_args.query_instruction_format_for_rerank,
            passage_instruction_for_rerank=model_args.passage_instruction_for_rerank,
            passage_instruction_format=model_args.passage_instruction_format_for_rerank,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
            devices=model_args.devices,
            normalize=model_args.normalize,
            prompt=model_args.prompt,
            cutoff_layers=model_args.cutoff_layers,
            compress_layers=model_args.compress_layers,
            compress_ratio=model_args.compress_ratio,
        )
    return retriever, reranker

def main():
    parser = HfArgumentParser([AbsModelArgs, BEIREvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: AbsModelArgs
    eval_args: BEIREvalArgs

    retriever, reranker = get_models(model_args)

    task_types = eval_args.task_types
    tasks = eval_args.tasks
    languages = eval_args.languages
    tasks = mteb.get_tasks(
        languages=languages,
        tasks=tasks,
        task_types=task_types
    )
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(retriever, output_folder=f"results/{str(retriever)}")

    # all_pairs = []
    # for task_type in eval_args.task_types:
    #     if task_type in tasks_desc.keys():
    #         for task_name in tasks_desc[task_type]:
    #             all_pairs.append((task_type, task_name))
    # for task_type in tasks_desc.keys():
    #     for v in tasks_desc[task_type]:
    #         if v in eval_args.task_types:
    #             all_pairs.append((task_type, v))
    # all_pairs = list(set(all_pairs))

if __name__ == "__main__":
    main()