import os
import torch
import logging
import datasets
from typing import List
from accelerate import Accelerator
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict

from src.retrieval import (
    RetrievalArgs, 
    Retriever, 
    RetrievalDataset, 
    RetrievalMetric,
    TASK_CONFIG,
)
from src.utils.util import makedirs, FileLogger

logger = logging.getLogger(__name__)


@dataclass
class Args(RetrievalArgs):
    eval_data: str = field(
        default=None,
        metadata={'help': 'Query jsonl.'}
    )
    output_dir: str = field(
        default="data/outputs/",
    )
    corpus: str = field(
        default=None,
        metadata={'help': 'Corpus path for retrieval.'}
    )
    key_template: str = field(
        default="{title} {text}",
        metadata={'help': 'How to concatenate columns in the corpus to form one key?'}
    )
    log_path: str = field(
        default="data/results/performance.log",
        metadata={'help': 'Path to the file for logging.'}
    )


def main(args, accelerator=None, log=True):
    if accelerator is None:
        accelerator = Accelerator(cpu=args.cpu)

    with accelerator.main_process_first():
        config = TASK_CONFIG[args.version]
        instruction = config["instruction"]

        # we should get the evaluation task before specifying instruction
        # NOTE: only dense retrieval needs instruction
        if args.eval_data is not None and args.add_instruction and args.retrieval_method == "dense":
            raw_eval_dataset = datasets.load_dataset('json', data_files=args.eval_data, split='train', cache_dir=args.dataset_cache_dir)
            eval_task = raw_eval_dataset[0]["task"]
        else:
            eval_task = None

        eval_dataset = RetrievalDataset.prepare_eval_dataset(
            data_file=args.eval_data, 
            cache_dir=args.dataset_cache_dir,
            instruction=instruction[eval_task] if eval_task is not None else None,
        )
        corpus = RetrievalDataset.prepare_corpus(
            data_file=args.corpus,
            key_template=args.key_template,
            cache_dir=args.dataset_cache_dir,
            instruction=instruction[eval_task] if eval_task is not None else None 
        )
    
    result_path = RetrievalMetric._get_save_path(args.eval_data, args.output_dir, field="result", save_name=args.save_name)

    if args.load_result:
        query_ids, preds = RetrievalMetric._load_result(result_path)
        
    else:
        retriever = Retriever(
            retrieval_method=args.retrieval_method,
            # for dense retriever
            query_encoder=args.query_encoder,
            key_encoder=args.key_encoder,
            pooling_method=args.pooling_method,
            dense_metric=args.dense_metric,
            query_max_length=args.query_max_length,
            key_max_length=args.key_max_length,
            tie_encoders=args.tie_encoders,
            truncation_side=args.truncation_side,
            cache_dir=args.model_cache_dir, 
            dtype=args.dtype,
            accelerator=accelerator,
            # for bm25 retriever
            anserini_dir=args.anserini_dir,
            k1=args.k1,
            b=args.b
        )

        retriever.index(
            corpus, 
            output_dir=args.output_dir, 
            # for dense retriever
            embedding_name=args.embedding_name,
            index_factory=args.faiss_index_factory,
            load_encode=args.load_encode,
            save_encode=args.save_encode,
            load_index=args.load_index, 
            save_index=args.save_index,
            batch_size=args.batch_size,
            # for bm25 retriever
            threads=args.threads, 
            language=args.language, 
            storeDocvectors=args.storeDocvectors,
            load_collection=args.load_collection,
        )

        query_ids, preds = retriever.search(
            eval_dataset=eval_dataset,
            hits=args.hits,
            # for dense retriever
            batch_size=args.batch_size,
        )
        
        del retriever
        torch.cuda.empty_cache()
        
        if args.save_result and accelerator.process_index == 0:
            RetrievalMetric._save_result(query_ids, preds, result_path)

    if accelerator.process_index == 0:
        # NOTE: this corpus is for computing metrics, where no instruction is given
        no_instruction_corpus = RetrievalDataset.prepare_corpus(
            data_file=args.corpus,
            key_template=args.key_template,
            cache_dir=args.dataset_cache_dir,
        )
        
        metrics = RetrievalMetric.get_metric_fn(
            args.metrics, 
            cutoffs=args.cutoffs, 
            eval_data=args.eval_data,
            corpus=no_instruction_corpus,
            save_name=args.save_name,
            output_dir=args.output_dir,
            save_to_output=args.save_to_output,
            max_neg_num=args.max_neg_num,
            cache_dir=args.dataset_cache_dir,
            filter_answers=args.filter_answers,
        )(query_ids, preds)

        if log:
            file_logger = FileLogger(makedirs(args.log_path))
            file_logger.log(metrics, Args=asdict(args))
    else:
        metrics = {}

    accelerator.wait_for_everyone()
    return query_ids, preds, metrics

if __name__ == "__main__":
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()
    main(args)
