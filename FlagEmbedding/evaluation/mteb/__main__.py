from transformers import HfArgumentParser

from FlagEmbedding import FlagAutoModel, FlagAutoReranker
from FlagEmbedding.abc.evaluation import AbsModelArgs, AbsEmbedder, AbsReranker, AbsEvaluator


from utils.arguments import MSMARCOEvalArgs
from utils.data_loader import MSMARCODataLoader


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