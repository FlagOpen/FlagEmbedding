import os
import logging
import datasets

from copy import deepcopy
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict

from src.lm import SRLMArgs, SelfRetrievalLM
from src.retrieval import Retriever, RetrievalArgs, TASK_CONFIG
from src.utils.util import makedirs, remove_eos, DefaultDataCollator, DatasetProcessFn, FileLogger

logger = logging.getLogger(__name__)
import transformers
# disable too long input warning
transformers.logging.set_verbosity_error()


# merge two args to get unified arguments
@dataclass
class LRLMArgs(RetrievalArgs, SRLMArgs):
    eval_data: str = field(
        default="llm-embedder:lrlm/books3/test.json",
        metadata={'help': 'Evaluation json file.'},
    )
    lm_batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation json file.'},
    )

    context_max_length: int = field(
        default=32768,
        metadata={'help': 'Evaluation json file.'},
    )
    anchor_length: int = field(
        default=160000,
        metadata={'help': 'Evaluation file containing long texts.'}
    )
    chunk_size: int = field(
        default=128,
        metadata={'help': 'How many tokens in a chunk?'}
    )
    key_num: int = field(
        default=8,
        metadata={'help': 'How many chunks to retrieve at a time?'}
    )
    chunk_batch_size: int = field(
        default=1,
        metadata={'help': 'How many retrieval & generation to execute in parallel?'}  
    )

    log_path: str = field(
        default="data/results/lrlm",
        metadata={'help': 'Path to the file for logging.'}
    )
    debug_retrieval: bool = field(
        default=False,
        metadata={'help': 'Check retrieval queries and values?'}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.retrieval_method == "bm25":
            # NOTE: we can only use naive bm25 for self retrieval
            self.retrieval_method = "naive-bm25"


def process_lrlm(tokenizer, context_max_length=4096, target_length=1024, anchor_length=160000):
    test = tokenizer("test", return_special_tokens_mask=True)["special_tokens_mask"]
    has_eos = False
    if test[-1] == 1:
        has_eos = True

    left_truncation_tokenizer = deepcopy(tokenizer)
    left_truncation_tokenizer.truncation_side = "left"

    @DatasetProcessFn()
    def _process(text, **kwds):
        output = {}
        text = text[:anchor_length]

        inputs = left_truncation_tokenizer(text, max_length=context_max_length, truncation=True, return_token_type_ids=False, add_special_tokens=False)

        if len(inputs.input_ids) < target_length:
            return None

        labels = inputs["input_ids"].copy()
        inputs_length = len(labels)
        labels[:-target_length] = [-100 for _ in range(inputs_length - target_length)]
        inputs["labels"] = labels

        for k, v in inputs.items():
            output[k] = v
        return output
    return _process


def main():
    parser = HfArgumentParser([LRLMArgs])
    args, = parser.parse_args_into_dataclasses()
    
    accelerator = Accelerator(cpu=args.cpu)

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
    
    if args.add_instruction:
        instruction = TASK_CONFIG[args.version]["instruction"]["lrlm"]
    else:
        instruction = None

    srlm = SelfRetrievalLM(
        model_name_or_path=args.model_name_or_path,
        retriever=retriever,
        dtype=args.lm_dtype,
        device_map=args.lm_device_map,
        padding_side=args.padding_side,
        cache_dir=args.model_cache_dir,
        context_window_size=args.context_window_size,
        chunk_size=args.chunk_size,
        key_num=args.key_num,
        chunk_batch_size=args.chunk_batch_size,
        add_key_continuation=args.add_key_continuation,
        retrieval_method=args.retrieval_method,
        order_method=args.order_method,
        integrate_method=args.integrate_method,
        instruction=instruction,
        debug_retrieval=args.debug_retrieval,
        add_sep=args.add_sep,
        accelerator=accelerator,
    )

    tokenizer = srlm.tokenizer

    logging.info(f"Loading data from {args.eval_data}...")

    if args.retrieval_method == "no" and args.context_max_length != args.context_window_size:
        logger.warning(f"Found retrieval_method is 'no', setting context_max_length to the same as context_window_size ({args.context_window_size})!")
        args.context_max_length = args.context_window_size

    with accelerator.main_process_first():
        dataset = datasets.load_dataset("json", data_files=args.eval_data, split="train", cache_dir=args.dataset_cache_dir)
        dataset = dataset.map(process_lrlm(
            tokenizer, 
            context_max_length=args.context_max_length,
            target_length=args.target_length,
            anchor_length=args.anchor_length,
        ), remove_columns=dataset.column_names, batched=True, batch_size=50, num_proc=64)
        
    data_collator = DefaultDataCollator(tokenizer=tokenizer, add_position_ids=args.add_position_ids)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.lm_batch_size, 
        collate_fn=data_collator,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)

    perplexity = srlm.compute_perplexity(dataloader)
    metrics = {"perplexity": perplexity}

    if accelerator.process_index == 0:
        dataset = os.path.normpath(args.eval_data).split(os.sep)[-2]
        log_path = os.path.join(args.log_path, f"{dataset}.log")

        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
