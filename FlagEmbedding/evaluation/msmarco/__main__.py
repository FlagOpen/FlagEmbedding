from transformers import HfArgumentParser

from FlagEmbedding import AutoFlagModel, AutoFlagReranker
from FlagEmbedding.abc.evaluation import AbsModelArgs, AbsRetriever, AbsReranker, AbsEvaluator


from utils.arguments import MSMARCOEvalArgs
from utils.data_loader import MSMARCODataLoader


def get_models(model_args: AbsModelArgs):
    retriever = AutoFlagModel.from_finetuned(
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
        reranker = AutoFlagReranker.from_finetuned(
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
            devices=model_args.devices
        )
    return retriever, reranker


def main():
    parser = HfArgumentParser([AbsModelArgs, MSMARCOEvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: AbsModelArgs
    eval_args: MSMARCOEvalArgs

    retriever, reranker = get_models(model_args)
    
    data_loader = MSMARCODataLoader(
        dataset_dir = eval_args.dataset_dir,
        cache_dir = eval_args.cache_path,
        text_type = eval_args.text_type
    )

    evaluation = AbsEvaluator(
        data_loader=data_loader,
        splits=eval_args.splits,
        overwrite=eval_args.overwrite,
    )
    
    retriever = AbsRetriever(
        embedding_model, 
        search_top_k=eval_args.search_top_k,
    )
    
    if cross_encoder is not None:
        reranker = AbsReranker(
            cross_encoder,
            rerank_top_k=eval_args.rerank_top_k,
        )
    else:
        reranker = None
    
    evaluation(
        splits=eval_args.split,
        search_results_save_dir=eval_args.output_dir,
        retriever=retriever,
        reranker=reranker,
        corpus_embd_save_dir=eval_args.corpus_embd_save_dir,
    )


if __name__ == "__main__":
    main()