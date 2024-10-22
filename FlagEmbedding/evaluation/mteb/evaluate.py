from transformers import HfArgumentParser

from FlagEmbedding import AutoFlagModel

from utils.arguments import ModelArgs
from utils.models import SentenceTransformerEncoder, SentenceTransformerReranker
from utils.searcher import EmbeddingModelRetriever, CrossEncoderReranker


def get_models(model_args: ModelArgs):
    embedding_model = AutoFlagModel.from_finetuned(
        model_name_or_path,
        normalize_embeddings,
        use_fp16,
        query_instruction_for_retrieval,
        query_instruction_format,
        devices,
        examples_for_task,
        examples_instruction_format,
        trust_remote_code,
        cache_dir
    )
    cross_encoder = None
    if model_args.reranker is not None:
        cross_encoder = SentenceTransformerReranker(
            model_name_or_path,
            peft_path,
            use_fp16,
            use_bf16,
            query_instruction_for_rerank,
            query_instruction_format,
            passage_instruction_for_rerank,
            passage_instruction_format,
            cache_dir,
            trust_remote_code,
            devices
        )
    return embedding_model, cross_encoder


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs

    embedding_model, cross_encoder = get_models(model_args)
    
    evaluation = AIRBench(
        benchmark_version=eval_args.benchmark_version,
        task_types=eval_args.task_types,
        domains=eval_args.domains,
        languages=eval_args.languages,
        splits=eval_args.splits,
        cache_dir=eval_args.cache_dir,
    )
    
    retriever = EmbeddingModelRetriever(
        embedding_model, 
        search_top_k=eval_args.search_top_k,
        corpus_chunk_size=model_args.corpus_chunk_size,
    )
    
    if cross_encoder is not None:
        reranker = CrossEncoderReranker(
            cross_encoder,
            rerank_top_k=eval_args.rerank_top_k,
        )
    else:
        reranker = None
    
    evaluation.run(
        retriever,
        reranker=reranker,
        output_dir=eval_args.output_dir,
        overwrite=eval_args.overwrite,
    )


if __name__ == "__main__":
    main()