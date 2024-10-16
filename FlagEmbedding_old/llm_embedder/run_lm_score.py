import os
import json
import logging
import random
import datasets
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from collections import defaultdict
from transformers import HfArgumentParser
from src.lm import LM, LMArgs
from src.utils.util import split_file_dir_name_ext, makedirs, save_pickle, load_pickle, remove_eos, DefaultDataCollator, DatasetProcessFn

logger = logging.getLogger(__name__)


@dataclass
class ScoreArgs(LMArgs):
    eval_data: str = field(
        default=None,
        metadata={'help': 'Query jsonl.'}
    )
    context_max_length: int = field(
        default=1024,
        metadata={'help': 'Max length for lm.'}
    )
    key_max_length: int = field(
        default=512,
        metadata={'help': 'Max length for key.'}
    )
    lm_batch_size: int = field(
        default=4,
        metadata={'help': 'Evaluation json file.'},
    )
    save_name: str = field(
        default="llama2-7b-chat",
        metadata={'help': 'Name of the scored file.'}
    )
    load_score: bool = field(
        default=False,
        metadata={'help': 'Load score from temperary file?'}
    )


def process_lm_scoring(tokenizer, key_max_length=512):
    test = tokenizer("test", return_special_tokens_mask=True)["special_tokens_mask"]
    has_bos = has_eos = False
    if test[0] == 1:
        has_bos = True
    if test[-1] == 1:
        has_eos = True

    @DatasetProcessFn(augment=True)
    def _process(query, answers, query_id, task, pos=None, neg=None, history=None, context_inputs=None, query_inputs=None, answer_inputs=None, score_inputs=None, _index=None, **kwds):
        """Yield each key (pos&neg)"""
        if task in ["qa", "convsearch"]:
            template = "Knowledge: {key.strip()}\n\nQuestion: {query.strip()}\n\nAnswer: {answer.strip()}"
        elif task == "icl":
            template = "{key}\n{query}\n{answer}"
        elif task == "lrlm":
            # template = "{key}{continuation[i]}{context}{query}{answer}"
            pass
        elif task == "chat":
            template = "{key}\nSpeaker 1: {query}\nSpeaker 2: {answer}"
        else:
            raise NotImplementedError(f"Task type {task} not implemented!")

        output = defaultdict(list)
        # NOTE: sample 1 answer for scoring if there are multiple
        if len(answers) > 1:
            answer = random.choice(answers)
        else:
            answer = answers[0]

        if history is not None:
            assert task == "chat", f"Found history={history} is not None but task={task} is not 'chat'!"
            keys = history
        else:
            keys = pos + neg
        for i, key in enumerate(keys):
            # NOTE: do not add special tokens!
            if task == "lrlm":
                score_input = score_inputs[i]
                input_ids = score_input + context_inputs + query_inputs + answer_inputs
                attention_mask = [1 for _ in input_ids]
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                labels = input_ids.copy()
                answer_length = len(answer_inputs)
                labels[:-answer_length] = [-100] * (len(labels) - answer_length)
                inputs["labels"] = labels
            else:
                # truncate key
                key = tokenizer.decode(tokenizer.encode(key, add_special_tokens=False, max_length=key_max_length, truncation=True))

                seq = eval(f"f{repr(template)}")
                inputs = tokenizer(seq, return_token_type_ids=False)
                if has_eos:
                    inputs = remove_eos(inputs, tokenizer.eos_token_id)

                # find answer length
                answer_seq = tokenizer.encode("Answer: " + answer.lstrip(" "), add_special_tokens=False)
                answer_length = len(answer_seq) - len(tokenizer.encode("Answer:", add_special_tokens=False))
                assert answer_length > 0, f"No answer found in inputs {_index}!"

                # take care of padded tokens
                labels = inputs["input_ids"].copy()
                labels = [x if inputs["attention_mask"][i] == 1 else -100 for i, x in enumerate(labels)]
                labels[:-answer_length] = [-100] * (len(labels) - answer_length)
                inputs["labels"] = labels

            for k, v in inputs.items():
                output[k].append(v)
            output["query_id"].append(query_id)
        return output
    return _process


def collate_scores(eval_data, save_name):
    """
    Collate the lm scorings based on query_ids. 
    Append a 'teacher_score' column in the eval_data and save at eval_data.save_name.json.
    """
    def collate(query_ids, scores):
        # only on main process
        eval_data_folder, eval_data_name, eval_data_ext = split_file_dir_name_ext(eval_data)
        data_save_path = os.path.join(eval_data_folder, f"{eval_data_name}.scored.{save_name}" + eval_data_ext)
        makedirs(data_save_path)

        prev_query_id = None
        teacher_scores = []
        try:
            logger.info(f"saving data to {data_save_path}...")
            with open(eval_data) as f, open(data_save_path, "w") as g:
                for query_id, score in tqdm(zip(query_ids, scores)):
                    if (query_id != prev_query_id) and (prev_query_id is not None):
                        sample = json.loads(f.readline().strip())
                        assert prev_query_id == sample["query_id"], f"Found incompatible query_id from data ({sample['query_id']}) and from eval_preds ({prev_query_id})"
                        if "history" in sample:
                            assert len(sample["history"]) == len(teacher_scores), f"Found incompatible key number from data ({len(sample['history'])}) and from eval_preds ({len(teacher_scores)})"
                        else:
                            assert len(sample["pos"] + sample["neg"]) == len(teacher_scores), f"Found incompatible key number from data ({len(sample['pos'] + sample['neg'])}) and from eval_preds ({len(teacher_scores)})"
                        sample["teacher_scores"] = teacher_scores.copy()
                        if sample["task"] == "lrlm" and "query_inputs" in sample:
                            del sample["query_inputs"]
                            del sample["answer_inputs"]
                            del sample["context_inputs"]
                            del sample["score_inputs"]

                        g.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        teacher_scores.clear()
                        
                    # accumulate scores of different keys for the same query
                    # log likelihood
                    teacher_scores.append(-score)
                    prev_query_id = query_id

                # NOTE: the last line
                sample = json.loads(f.readline().strip())
                assert prev_query_id == sample["query_id"], f"Found incompatible query_id from data ({sample['query_id']}) and from eval_preds ({prev_query_id})"
                if "history" in sample:
                    assert len(sample["history"]) == len(teacher_scores), f"Found incompatible key number from data ({len(sample['history'])}) and from eval_preds ({len(teacher_scores)})"
                else:
                    assert len(sample["pos"] + sample["neg"]) == len(teacher_scores), f"Found incompatible key number from data ({len(sample['pos'] + sample['neg'])}) and from eval_preds ({len(teacher_scores)})"
                sample["teacher_scores"] = teacher_scores.copy()
                if sample["task"] == "lrlm" and "query_inputs" in sample:
                    del sample["query_inputs"]
                    del sample["answer_inputs"]
                    del sample["context_inputs"]
                    del sample["score_inputs"]
                g.write(json.dumps(sample, ensure_ascii=False) + "\n")
                teacher_scores.clear()

        except:
            save_path = os.path.join(eval_data_folder, f"{eval_data_name}.{save_name}.pkl")
            logger.error(f"Error when trying to save to json file. Save scores to {save_path} instead!")
            save_pickle((query_ids, scores), save_path)
            raise
    return collate


def main():
    parser = HfArgumentParser([ScoreArgs])
    args, = parser.parse_args_into_dataclasses()
    args: ScoreArgs
    
    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100000))])
    logger.info(f"Loading data from {args.eval_data}...")

    llm = LM(
        model_name_or_path=args.model_name_or_path,
        dtype=args.lm_dtype,
        padding_side=args.padding_side,
        cache_dir=args.model_cache_dir,
        accelerator=accelerator
    )
    llm.to(accelerator.device)

    tokenizer = llm.tokenizer

    logging.info(f"Loading data from {args.eval_data}...")

    if args.load_score:
        eval_data_folder, eval_data_name, eval_data_ext = split_file_dir_name_ext(args.eval_data)
        save_path = os.path.join(eval_data_folder, f"{eval_data_name}.{args.save_name}.pkl")
        results = load_pickle(save_path)

    else:
        with accelerator.main_process_first():
            # dataset = datasets.load_dataset("json", data_files=args.eval_data, split="train[:100]", cache_dir=args.dataset_cache_dir)
            dataset = datasets.load_dataset("json", data_files=args.eval_data, split="train", cache_dir=args.dataset_cache_dir)
            dataset = dataset.map(
                process_lm_scoring(tokenizer=tokenizer, key_max_length=args.key_max_length), 
                remove_columns=dataset.column_names, 
                batched=True, 
                num_proc=32, 
                with_indices=True
            )

        data_collator = DefaultDataCollator(tokenizer=tokenizer, add_position_ids=args.add_position_ids)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.lm_batch_size, 
            collate_fn=data_collator,
            pin_memory=True,
        )
        dataloader = accelerator.prepare(dataloader)
        
        query_ids, scores = llm.compute_nlls(dataloader)

    if accelerator.process_index == 0:
        collate_scores(args.eval_data, args.save_name)(query_ids, scores)


if __name__ == "__main__":
    main()
