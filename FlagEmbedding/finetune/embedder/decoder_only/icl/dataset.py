import math
import random
import logging
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
)

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderSameDatasetTrainDataset

from .arguments import DecoderOnlyEmbedderICLDataArguments

logger = logging.getLogger(__name__)


class DecoderOnlyEmbedderICLSameDatasetTrainDataset(AbsEmbedderSameDatasetTrainDataset):
    """Dataset class for icl model.

    Args:
        args (DecoderOnlyEmbedderICLDataArguments): Data argument class for icl model.
        default_batch_size (int): The default batch size.
        seed (int): Random seed to use.
        tokenizer (PreTrainedTokenizer): Tokenzier.
        process_index (int, optional): Current process index. Defaults to 0.
        num_processes (int, optional): Total number of processes. Defaults to 1.
    """
    def __init__(
        self,
        args: DecoderOnlyEmbedderICLDataArguments,
        default_batch_size: int,
        seed: int,
        tokenizer: PreTrainedTokenizer,
        process_index: int=0,
        num_processes: int=1
    ):
        super().__init__(
            args=args,
            default_batch_size=default_batch_size,
            seed=seed,
            tokenizer=tokenizer,
            process_index=process_index,
            num_processes=num_processes
        )
        self.args: DecoderOnlyEmbedderICLDataArguments

        self.suffix = self.tokenizer(f"{self.args.icl_suffix_str}{self.tokenizer.eos_token}", add_special_tokens=False)['input_ids']

        self.prefix = self.tokenizer(f"{self.tokenizer.bos_token}", add_special_tokens=False)['input_ids']

    def _create_batch_data(self, batch_raw_data):
        """Create a comple batch of data with queries, documents and teacher scores.

        Args:
            batch_raw_data (datasets.Dataset): One batch of raw data.

        Returns:
            List[str]: Queries with instruction format.
            List[str]: Documents with instruction format.
            List[float]: Teacher scores for model distillation.
        """
        queries, passages, teacher_scores = [], [], []

        train_group_size, data_type = self._get_train_group_size(batch_raw_data)

        icl_pairs = []

        for i in range(len(batch_raw_data['query'])):
            if data_type is not None:
                assert batch_raw_data['type'][i] == data_type, f"Data type is not consistent in the same batch"

            queries.append(
                self.args.query_instruction_format.format(
                    batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                    batch_raw_data['query'][i]
                )
            )
            tmp_passages = []
            pos_idx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            pos = self._shuffle_text(batch_raw_data['pos'][i][pos_idx])
            tmp_passages.append(pos)

            neg_all_idx = list(range(len(batch_raw_data['neg'][i])))
            if len(batch_raw_data['neg'][i]) < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
            else:
                neg_idxs = random.sample(neg_all_idx, train_group_size - 1)
            for neg_idx in neg_idxs:
                tmp_passages.append(batch_raw_data['neg'][i][neg_idx])

            if self.args.knowledge_distillation:
                if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                    teacher_scores.append(batch_raw_data['pos_scores'][i][pos_idx])
                for neg_idx in neg_idxs:
                    if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
                        teacher_scores.append(batch_raw_data['neg_scores'][i][neg_idx])
            else:
                teacher_scores = None

            if data_type is not None and data_type in ['symmetric_sts', 'symmetric_clustering']:
                tmp_passages = [
                    self.args.query_instruction_format.format(
                        batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                        p
                    ) for p in tmp_passages
                ]
                tmp_passages = self.tokenizer.batch_decode(
                    self.tokenizer(
                        tmp_passages,
                        max_length=self.args.passage_max_len - 1 - len(self.suffix),
                        truncation=True,
                        add_special_tokens=False,
                    )['input_ids']
                )
                for j in range(len(tmp_passages)):
                    tmp_passages[j] += self.args.icl_suffix_str
            else:
                if self.args.passage_instruction_for_retrieval is not None:
                    tmp_passages = [
                        self.args.passage_instruction_format.format(
                            self.args.passage_instruction_for_retrieval, p
                        ) for p in tmp_passages
                    ]

            passages.extend(tmp_passages)
            
            if len(teacher_scores) > 0 and len(passages) > 0:
                assert len(teacher_scores) == len(passages)

            # add icl pairs
            if self.args.retrieval_use_examples or (
                data_type in ['symmetric_sts', 'symmetric_clustering', 'symmetric_class']
            ):
                if data_type == 'symmetric_clustering':
                    icl_pairs.append((
                        self.tokenizer.decode(
                            self.tokenizer(
                                queries[-1],
                                add_special_tokens=False
                            )['input_ids'][:self.args.example_query_max_len]
                        ),
                        self.tokenizer.decode(
                            self.tokenizer(
                                batch_raw_data['category'][i],  # use category as example
                                add_special_tokens=False
                            )['input_ids'][:self.args.example_passage_max_len]
                        )
                    ))
                else:
                    icl_pairs.append((
                        self.tokenizer.decode(
                            self.tokenizer(
                                queries[-1],
                                add_special_tokens=False
                            )['input_ids'][:self.args.example_query_max_len]
                        ),
                        self.tokenizer.decode(
                            self.tokenizer(
                                pos,
                                add_special_tokens=False
                            )['input_ids'][:self.args.example_passage_max_len]
                        )
                    ))
            else:
                icl_pairs = []

        # handle queries
        for i in range(len(queries)):
            choices = random.choice([0, 1, 2, 3, 4, 5])
            if choices > 0 and len(icl_pairs) > 0:
                prefix_ids = random.sample(list(range(len(icl_pairs))), min(choices + 1, len(icl_pairs)))
                if i in prefix_ids:
                    prefix_ids.remove(i)
                prefix_ids = prefix_ids[:choices]
                
                prefix = ''
                for idx in prefix_ids:
                    tmp = prefix + self.args.icl_suffix_str.join(icl_pairs[idx]) + '\n\n'
                    if len(self.tokenizer(tmp)['input_ids']) > self.args.query_max_len - 512:
                        break
                    prefix = tmp
            else:
                prefix = ''

            queries[i] = prefix + queries[i]
            queries[i] = self.tokenizer.decode(
                self.tokenizer(
                    queries[i],
                    max_length=self.args.query_max_len - len(self.prefix) - len(self.suffix),
                    truncation=True,
                    add_special_tokens=False
                )['input_ids']
            ) + self.args.icl_suffix_str

        return queries, passages, teacher_scores


@dataclass
class AbsEmbedderSameDatasetCollator(DataCollatorWithPadding):
    """
    EmbedCollator for SameDataset.
    Note that after using this collator, the training_args should be set as:
    
    ``training_args.per_device_train_batch_size = 1``
    
    ``training_args.dataloader_num_workers = 0    # avoid multi-processing``
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1

    def __call__(self, features):
        queries = features[0][0]
        passages = features[0][1]
        teacher_scores = features[0][2]
        no_in_batch_neg_flag = features[0][3]
        
        queries_inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        passages_inputs = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            d_collated = self.tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(queries_inputs['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries_inputs.items():
                    sub_features[k] = v[start:end]
                q_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

            d_collated = []
            for i in range(0, len(passages_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(passages_inputs['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages_inputs.items():
                    sub_features[k] = v[start:end]
                d_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": no_in_batch_neg_flag
        }
