import os
import math
import random
import logging
import datasets
import numpy as np
import torch.distributed as dist
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderDataArguments, AbsEmbedderTrainingArguments
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainDataset, AbsEmbedderCollator, AbsEmbedderSameDatasetTrainDataset, AbsEmbedderSameDatasetCollator, EmbedderTrainerCallbackForDataRefresh

logger = logging.getLogger(__name__)


class IREmbedderTrainDataset(AbsEmbedderTrainDataset):
    """Abstract class for training dataset.

    Args:
        args (AbsEmbedderDataArguments): Data arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
    """
    def __init__(
        self,
        args: AbsEmbedderDataArguments,
        tokenizer: PreTrainedTokenizer
    ):
        super().__init__(
            args,
            tokenizer,
        )

    def __getitem__(self, item):
        data = self.dataset[item]
        train_group_size = self.args.train_group_size

        query = data['query']
        answer = data.get('answer', None)
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_format.format(
                data['prompt'] if 'prompt' in data else self.args.query_instruction_for_retrieval,
                query
            )

        passages = []
        teacher_scores = []

        assert isinstance(data['pos'], list) and isinstance(data['neg'], list)

        pos_idx = random.choice(list(range(len(data['pos']))))
        passages.append(self._shuffle_text(data['pos'][pos_idx]))

        neg_all_idx = list(range(len(data['neg'])))
        if len(data['neg']) < train_group_size - 1:
            num = math.ceil((train_group_size - 1) / len(data['neg']))
            neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
        else:
            neg_idxs = random.sample(neg_all_idx, self.args.train_group_size - 1)
        for neg_idx in neg_idxs:
            passages.append(data['neg'][neg_idx])

        if self.args.knowledge_distillation:
            assert isinstance(data['pos_scores'], list) and isinstance(data['neg_scores'], list)
            teacher_scores.append(data['pos_scores'][pos_idx])
            for neg_idx in neg_idxs:
                teacher_scores.append(data['neg_scores'][neg_idx])
            if not all(isinstance(score, (int, float)) for score in teacher_scores):
                raise ValueError(f"pos_score or neg_score must be digit")
        else:
            teacher_scores = None

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [
                self.args.passage_instruction_format.format(
                    self.args.passage_instruction_for_retrieval, p
                )
                for p in passages
            ]

        return query, answer, passages, teacher_scores

@dataclass
class IREmbedderCollator(AbsEmbedderCollator):
    """
    The abstract embedder collator.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1

    def __call__(self, features):
        queries = [f[0] for f in features]
        answers = [f[1] for f in features]
        passages = [f[2] for f in features]
        teacher_scores = [f[3] for f in features]
        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])

        if isinstance(queries[0], list):
            queries = sum(queries, [])
        if isinstance(answers[0], list):
            answers = sum(answers, [])
        if isinstance(passages[0], list):
            passages = sum(passages, [])

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
        if answers[0] is None and answers[-1] is None:
            answers_inputs = self.tokenizer(
                answers,
                truncation=True,
                max_length=self.query_max_len,
                return_tensors=None
            )
        else:
            answers_inputs = None

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
            d_collated = self.tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
            if answers_inputs is None:
                a_collated = None
            else:
                a_collated = self.tokenizer.pad(
                    answers_inputs,
                    padding=self.padding,
                    max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors
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
                    return_tensors=self.return_tensors
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
                    return_tensors=self.return_tensors
                ))
            
            if answers_inputs is None:
                a_collated = None
            else:
                a_collated = []
                for i in range(0, len(answers_inputs['attention_mask']), batch_size):
                    start = i
                    end = min(len(answers_inputs['attention_mask']), i + batch_size)
                    sub_features = {}
                    for k, v in answers_inputs.items():
                        sub_features[k] = v[start:end]
                    a_collated.append(self.tokenizer.pad(
                        sub_features,
                        padding=self.padding,
                        max_length=self.passage_max_len,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors
                    ))

        return {
            "queries": q_collated,
            "answers": a_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": False
        }


class IREmbedderSameDatasetTrainDataset(AbsEmbedderSameDatasetTrainDataset):
    """Abstract class for training dataset that samples batches from same dataset.

    Args:
        args (AbsEmbedderDataArguments): Data arguments.
        default_batch_size (int): The default batch size for training.
        seed (int): Random seed.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        process_index (int, optional): Current process index. Defaults to 0.
        num_processes (int, optional): Total number of processes. Defaults to 1.
    """
    def __init__(
        self,
        args: AbsEmbedderDataArguments,
        default_batch_size: int,
        seed: int,
        tokenizer: PreTrainedTokenizer,
        process_index: int=0,
        num_processes: int=1
    ):
        super().__init__(
            args,
            default_batch_size,
            seed,
            tokenizer,
            process_index,
            num_processes
        )
    
    def _shuffle_answer(self, text):
        """shuffle the input text.

        Args:
            text (str): Input text.

        Returns:
            str: Shuffled text.
        """
        split_text = []
        chunk_size = len(text)//3 + 1
        for i in range(0, len(text), chunk_size):
            split_text.append(text[i:i+chunk_size])
        random.shuffle(split_text)
        return " ".join(split_text)

    def __getitem__(self, _):
        batch_indices, no_in_batch_neg_flag = self.batch_datas[self.step]    # extend here
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries, answers, passages, teacher_scores, teacher_scores_answers = self._create_batch_data(batch_raw_data=batch_data)
        return queries, answers, passages, teacher_scores, teacher_scores_answers, no_in_batch_neg_flag
    def _create_batch_data(self, batch_raw_data):
        """Create a comple batch of data with queries, documents and teacher scores.

        Args:
            batch_raw_data (datasets.Dataset): One batch of raw data.

        Returns:
            List[str]: Queries with instruction format.
            List[str]: Documents with instruction format.
            List[float]: Teacher scores for model distillation.
        """
        queries, answers, passages, teacher_scores, teacher_scores_answers = [], [], [], [], []

        train_group_size, data_type = self._get_train_group_size(batch_raw_data)

        for i in range(len(batch_raw_data['query'])):
            if data_type is not None:
                assert batch_raw_data['type'][i] == data_type, f"Data type is not consistent in the same batch"

            queries.append(
                self.args.query_instruction_format.format(
                    batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                    batch_raw_data['query'][i]
                )
            )
            if 'answer' in batch_raw_data.keys():
                answers.append(
                    self.args.query_instruction_format.format(
                        batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                        batch_raw_data['answer'][i]
                    )
                )
                # answers[-1] = self._shuffle_answer(answers[-1])
            else:
                answers.append(None)

            tmp_passages = []
            pos_idx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            pos = self._shuffle_text(batch_raw_data['pos'][i][pos_idx])
            # pos = self._shuffle_answer(batch_raw_data['answer'][i])
            # pos = batch_raw_data['answer'][i]
            tmp_passages.append(pos)

            if train_group_size == 1:
                pass
            else:
                neg_all_idx = list(range(len(batch_raw_data['neg'][i])))
                if len(batch_raw_data['neg'][i]) < train_group_size - 1:
                    num = math.ceil((train_group_size - 1) / len(batch_raw_data['neg'][i]))
                    neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
                else:
                    neg_idxs = random.sample(neg_all_idx, train_group_size - 1)
                
                if self.args.knowledge_distillation:
                    tmp_scores = [batch_raw_data['neg_scores'][i][neg_idx] for neg_idx in neg_idxs]
                    tmp_data = sorted([(x, y) for x, y in zip(neg_idxs, tmp_scores)], reverse=True, key=lambda x: x[1])
                    neg_idxs = [x[0] for x in tmp_data]

                for neg_idx in neg_idxs:
                    tmp_passages.append(batch_raw_data['neg'][i][neg_idx])
                    # answers.append(batch_raw_data['neg'][i][neg_idx])

            if self.args.knowledge_distillation:
                if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                    if batch_raw_data['pos_scores'][i][pos_idx] < max(batch_raw_data['neg_scores'][i]):
                        teacher_scores.append(batch_raw_data['pos_scores'][i][pos_idx])
                    else:
                        teacher_scores.append(
                            batch_raw_data['pos_scores'][i][pos_idx] + 
                            (max(batch_raw_data['neg_scores'][i]) - batch_raw_data['pos_scores'][i][pos_idx]) * 0.2
                        )
                for neg_idx in neg_idxs:
                    if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
                        teacher_scores.append(batch_raw_data['neg_scores'][i][neg_idx])
            else:
                teacher_scores = None
            
            ### add answer knowledge distillation
            if self.args.answer_inbatch:
                if train_group_size == 1:
                    pass
                else:
                    neg_all_idx = list(range(len(batch_raw_data['neg_answer'][i])))
                    if len(batch_raw_data['neg_answer'][i]) < train_group_size - 1:
                        num = math.ceil((train_group_size - 1) / len(batch_raw_data['neg_answer'][i]))
                        neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
                    else:
                        neg_idxs = random.sample(neg_all_idx, train_group_size - 1)
                    for neg_idx in neg_idxs:
                        answers.append(batch_raw_data['neg_answer'][i][neg_idx])
            # if self.args.knowledge_distillation:
            #     if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
            #         teacher_scores_answers.append(batch_raw_data['pos_scores'][i][pos_idx])
            #     for neg_idx in neg_idxs:
            #         if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
            #             teacher_scores_answers.append(batch_raw_data['neg_scores'][i][neg_idx])
            # else:
            #     teacher_scores_answers = None

            if data_type is not None and data_type in ['symmetric_sts', 'symmetric_clustering']:
                tmp_passages = [
                    self.args.query_instruction_format.format(
                        batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                        p
                    ) for p in tmp_passages
                ]
            else:
                if self.args.passage_instruction_for_retrieval is not None:
                    tmp_passages = [
                        self.args.passage_instruction_format.format(
                            self.args.passage_instruction_for_retrieval, p
                        ) for p in tmp_passages
                    ]

            passages.extend(tmp_passages)

            if teacher_scores is not None:
                if len(teacher_scores) > 0 and len(passages) > 0:
                    assert len(teacher_scores) == len(passages)

        return queries, answers, passages, teacher_scores, teacher_scores_answers


@dataclass
class IREmbedderSameDatasetCollator(AbsEmbedderSameDatasetCollator):
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
        answers = features[0][1]
        passages = features[0][2]
        teacher_scores = features[0][3]
        teacher_scores_answers = features[0][4]
        no_in_batch_neg_flag = features[0][5]

        queries_inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        if answers[0] is not None:
            answers_inputs = self.tokenizer(
                answers,
                truncation=True,
                max_length=self.query_max_len,
                return_tensors=None
            )
        else:
            answers_inputs = None
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

            if answers_inputs is None:
                a_collated = None
            else:
                a_collated = self.tokenizer.pad(
                    answers_inputs,
                    padding=self.padding,
                    max_length=self.query_max_len,
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
                    max_length=self.query_max_len,
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
            
            if answers_inputs is None:
                a_collated = None
            else:
                a_collated = []
                for i in range(0, len(answers_inputs['attention_mask']), batch_size):
                    start = i
                    end = min(len(answers_inputs['attention_mask']), i + batch_size)
                    sub_features = {}
                    for k, v in answers_inputs.items():
                        sub_features[k] = v[start:end]
                    a_collated.append(self.tokenizer.pad(
                        sub_features,
                        padding=self.padding,
                        max_length=self.query_max_len,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors=self.return_tensors,
                    ))

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "answers": a_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "teacher_scores_answers": teacher_scores_answers,
            "no_in_batch_neg_flag": no_in_batch_neg_flag
        }