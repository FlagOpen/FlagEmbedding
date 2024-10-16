import os
import json
import logging
import datasets
import random
from typing import List
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict

from src.lm import (
    LM, 
    LMArgs,
    GenerationArgs
)
from src.retrieval import (
    RetrievalArgs, 
    RetrievalMetric,
)
from src.utils.util import makedirs, remove_eos, normalize_text, DefaultDataCollator, DatasetProcessFn, FileLogger
from .eval_retrieval import main as retrieval_main
from .icl_utils import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class QRECCArgs(LMArgs, RetrievalArgs):
    output_dir: str = field(
        default="data/results/qrecc",
    )
    eval_data: str = field(
        default="llm-embedder:convsearch/qrecc/test.concat.json",
        metadata={'help': 'Query jsonl.'}
    )
    corpus: str = field(
        default="llm-embedder:convsearch/qrecc/corpus.json",
        metadata={'help': 'Corpus path for retrieval.'}
    )
    key_template: str = field(
        default="{text}",
        metadata={'help': 'How to concatenate columns in the corpus to form one key?'}
    )
    do_generate: bool = field(
        default=False,
        metadata={'help': 'Generate for computing qa metrics?'}
    )
    
    hits: int = field(
        default=100,
        metadata={'help': 'How many hits per query?'},
    )
    key_num: int = field(
        default=3,
        metadata={'help': 'How many docs to provide in prompt?'},
    )
    metrics: List[str] = field(
        default_factory=lambda: ["ndcg", "recall", "collate_key"],
    )
    cutoffs: List[int] = field(
        default_factory=lambda: [3, 10, 100],
        metadata={'help': 'Cutoffs to evaluate retrieval metrics.'}
    )
    max_neg_num: int = field(
        default=32,
        metadata={'help': 'Maximum negative number to mine.'}
    )
    save_to_output: bool = field(
        default=True,
        metadata={'help': 'Save the result/key/negative to output_dir? If not true, they will be saved next to the eval_data.'}
    )

    log_path: str = field(
        default="data/results/qrecc/qrecc.log",
        metadata={'help': 'Path to the file for logging.'}
    )


@dataclass
class GenerationArgs(GenerationArgs):
    max_new_tokens: int = field(
        default=128,
        metadata={'help': 'Maximum new tokens to generate.'}
    )
    eos_token_id: int = 13


def process_qrecc(tokenizer, context_max_length=2048, key_num=3, is_encoder_decoder=False):
    test = tokenizer("test", return_special_tokens_mask=True)["special_tokens_mask"]
    has_bos = has_eos = False
    if test[0] == 1:
        has_bos = True
    if test[-1] == 1:
        has_eos = True

    def _prepare_sample(query, answers=None, **kwds):
        sample = f"Context and Question: {query}\nAnswer:"
        if answers is not None:
            sample = sample + " " + random.choice(answers)
        return sample

    def _prepare_retrieval(keys):
        if keys is not None:
            keys = keys[:key_num]
            keys = "\n".join(keys)
            knowledge = f"Knowledge: {keys}"
        else:
            knowledge = ""
        return knowledge

    @DatasetProcessFn()
    def _process(query, query_id, key=None, **kwds):
        """Yield keys and query with a prompt template"""
        output = {}
        query = query.strip()
        knowledge = _prepare_retrieval(key)

        left = knowledge
        # \n\n to split retrieved knowledge
        right = "\n\n" + _prepare_sample(query)

        pair = tokenizer.encode(left, right, add_special_tokens=False, truncation="only_first", max_length=context_max_length - int(has_bos) - int(has_eos))

        # strip spaces and \n in the head (when there is no retrieved passage)
        seq = tokenizer.decode(pair).strip()
        inputs = tokenizer(seq, return_token_type_ids=False)

        if has_eos and not is_encoder_decoder:
            inputs = remove_eos(inputs, tokenizer.eos_token_id)

        inputs["query_id"] = query_id

        for k, v in inputs.items():
            output[k] = v
        return output
    return _process


def evaluate_qrecc(eval_data, save_path, **kwds):
    def compute_metric(eval_preds):
        makedirs(save_path)
        
        samples = {}
        with open(eval_data) as f:
            for line in f:
                sample = json.loads(line.strip())
                samples[sample["query_id"]] = sample["answers"][0]

        preds = []
        answers = []
        with open(save_path, "w") as f:
            for query_id, generation in zip(*eval_preds):
                answer = samples[query_id]
                preds.append(generation)
                answers.append(answer)

                sample["output"] = generation
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        rouge_l = compute_metrics("rl", labels=answers, preds=preds)
        return rouge_l
    return compute_metric


def main():
    parser = HfArgumentParser([QRECCArgs, GenerationArgs])
    args, generation_args = parser.parse_args_into_dataclasses()
    
    accelerator = Accelerator(cpu=args.cpu)
    
    # modify the output_dir for retrieval
    if args.retrieval_method == "dense":
        output_dir = os.path.join(args.output_dir, args.query_encoder.strip(os.sep).replace(os.sep, "--"))
    else:
        output_dir = os.path.join(args.output_dir, args.retrieval_method)
    args.output_dir = output_dir

    if args.retrieval_method != "no":
        # retrieval metrics computes ndcg and recall
        _, _, metrics = retrieval_main(args=args, accelerator=accelerator, log=False)
        eval_data = RetrievalMetric._get_save_path(args.eval_data, args.output_dir, field="key", save_name=args.save_name)
    else:
        eval_data = args.eval_data
        metrics = {}

    if args.do_generate:
        llm = LM(
            model_name_or_path=args.model_name_or_path,
            dtype=args.lm_dtype,
            device_map=args.lm_device_map,
            padding_side=args.padding_side,
            cache_dir=args.model_cache_dir,
            accelerator=accelerator,
            generation_args=asdict(generation_args)
        )

        tokenizer = llm.tokenizer
        
        logging.info(f"Loading data from {eval_data}...")

        with accelerator.main_process_first():
            dataset = datasets.load_dataset("json", data_files=eval_data, split="train", cache_dir=args.dataset_cache_dir)
            dataset = dataset.map(process_qrecc(
                tokenizer, 
                context_max_length=args.context_max_length, 
                key_num=args.key_num,
                is_encoder_decoder=llm.model.config.is_encoder_decoder
            ), remove_columns=dataset.column_names, batched=True, num_proc=32)

        data_collator = DefaultDataCollator(tokenizer=tokenizer, add_position_ids=args.add_position_ids)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.lm_batch_size, 
            collate_fn=data_collator,
            pin_memory=True,
        )
        dataloader = accelerator.prepare(dataloader)

        results = llm.generate(dataloader)
        if accelerator.process_index == 0:
            result_path = os.path.join(args.output_dir, args.model_name_or_path.strip(os.sep).replace(os.sep, "--") + ".json")
            lm_metrics = evaluate_qrecc(eval_data, result_path)(results)

    else:
        lm_metrics = {}

    if accelerator.process_index == 0:
        file_logger = FileLogger(makedirs(args.log_path))
        metrics.update(lm_metrics)
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
