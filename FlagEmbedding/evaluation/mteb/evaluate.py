import multiprocessing
import os
import random
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import json
import numpy as np
import argparse

import mteb

from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from typing import List
from mteb import MTEB

from utils import logger, pool, move_to_cuda, get_detailed_instruct, get_task_def_by_task_name_and_type, create_batch_dict, tasks_desc, create_batch_query_dict

parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
parser.add_argument('--task-types', nargs='+', default=[], help='task types to evaluate')
parser.add_argument('--output-dir', default='',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--model-name-or-path', default='tmp-outputs/',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--peft-name-or-path', default=None, type=str)
parser.add_argument('--tokenizer-path', default=None)
parser.add_argument('--embedding-path', default=None)
parser.add_argument('--special-token', default=False, type=bool, help='whether to use special token')
parser.add_argument('--zero-shot', default=False, type=bool, help='whether to use zero shot icl')
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--all-num', default=8, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--examples-dir', default='/share/chaofan/code/embedder/evaluate_for_icl/examples', type=str)
parser.add_argument('--eight-special-token', default=False, type=bool)
parser.add_argument('--passage-prompt', default=False, type=bool)
parser.add_argument('--pool-type', default='last', type=bool)

args = parser.parse_args()
base_name: str = args.model_name_or_path.split('/')[-1]
# if args.eight_special_token is True:
#     args.pool_type = 'last_eight'
# else:
#     args.pool_type = 'last'

logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
# assert args.pool_type in ['cls', 'avg', 'last', 'weightedavg', 'last_eight'], 'pool_type should be cls / avg / last'
os.makedirs(args.output_dir, exist_ok=True)
ALL_NUM = args.all_num

def find_last_index(lst, element):
    if not isinstance(element, list):
        try:
            reversed_list = lst[::-1]
            index_in_reversed = reversed_list.index(element)
            last_index = len(lst) - 1 - index_in_reversed
            return last_index
        except ValueError:
            return -1
    else:
        last_index = -1
        for i in range(len(lst) - len(element)):
            if lst[i:i + len(element)] == element:
                last_index = i
                break
        return last_index

print(args.special_token)
class DenseEncoder(torch.nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, use_cache=False)
        if args.peft_name_or_path is not None:
            peft_name_or_path = args.peft_name_or_path.split(',')
            if args.embedding_path is not None:
                self.encoder.set_input_embeddings(torch.load(args.embedding_path))
            elif args.special_token and os.path.exists(os.path.join(peft_name_or_path[-1], 'embedding', 'emb.pth')):
                self.encoder.set_input_embeddings(torch.load(os.path.join(peft_name_or_path[-1], 'embedding', 'emb.pth')))
            for peft_path in peft_name_or_path:
                self.encoder = PeftModel.from_pretrained(self.encoder, peft_path)
                self.encoder = self.encoder.merge_and_unload()

        if args.tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.l2_normalize = True
        self.prompt = None
        self.prefix = ''
        self.suffix = ''

        self.encoder.half()
        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.eight_special_token = args.eight_special_token
        if args.eight_special_token:
            self.special_tokens = [self.tokenizer.eos_token_id] * 7
        else:
            self.special_tokens = []

        self.passage_prompt=  args.passage_prompt

        self.batch_size = args.batch_size

        if args.special_token:
            self.index_type = 0
            self.index_start_locs = self.tokenizer('<query>', add_special_tokens=False)['input_ids'][0]
            self.index_end_locs = self.tokenizer('<response>', add_special_tokens=False)['input_ids'][0]
        else:
            self.index_type = 1
            self.index_start_locs = self.tokenizer('\nQuery:', add_special_tokens=False)['input_ids'][1:]
            self.index_end_locs = self.tokenizer('\nResponse:', add_special_tokens=False)['input_ids'][1:]

        # if self.gpu_count > 1:
        #     self.encoder = torch.nn.DataParallel(self.encoder)

    def get_loc(self, sentence):
        sentence = list(sentence)
        if isinstance(self.index_start_locs, int):
            return find_last_index(sentence, self.index_start_locs) + 1, find_last_index(sentence, self.index_end_locs)
        else:
            return find_last_index(sentence, self.index_start_locs) + len(self.index_start_locs), find_last_index(
                sentence, self.index_end_locs)

    @torch.no_grad()
    def encode(self, sentences, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        input_texts: List[str] = [self.prompt + s for s in sentences]

        encoded_embeds = []
        batch_size = self.batch_size * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_query_dict(self.tokenizer, self.prefix, self.suffix, batch_input_texts, special_tokens=self.special_tokens)
            # if self.device == 0:
            # print(self.tokenizer.decode(batch_dict['input_ids'][0]))
            # batch_dict = batch_dict.to(self.device)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                if args.pool_type == 'mean':
                    seq_locs = [self.get_loc(sentence) for sentence in batch_dict['input_ids']]
                    embeds = torch.stack(
                        [
                            outputs.last_hidden_state[i, start: end, :].mean(dim=0)
                            for i, (start, end) in enumerate(seq_locs)
                        ]
                    )
                else:
                    embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], 'last')
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    @torch.no_grad()
    def encode_queries(self, sentences, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        input_texts: List[str] = [self.prompt + s for s in sentences]

        encoded_embeds = []
        batch_size = self.batch_size * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_query_dict(self.tokenizer, self.prefix, self.suffix, batch_input_texts, special_tokens=self.special_tokens)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                if args.pool_type == 'mean':
                    seq_locs = [self.get_loc(sentence) for sentence in batch_dict['input_ids']]
                    embeds = torch.stack(
                        [
                            outputs.last_hidden_state[i, start: end, :].mean(dim=0)
                            for i, (start, end) in enumerate(seq_locs)
                        ]
                    )
                else:
                    embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], 'last')
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    @torch.no_grad()
    def encode_corpus(self, sentences, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        if isinstance(sentences[0], str):
            input_texts: List[str] = [s for s in sentences]
        else:
            input_texts: List[str] = [(s['text'] + ' ' + s['title']).strip() for s in sentences]

        encoded_embeds = []
        batch_size = self.batch_size * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, special_tokens=self.special_tokens, passage_prompt=self.passage_prompt)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                if args.pool_type == 'mean':
                    seq_locs = [self.get_loc(sentence) for sentence in batch_dict['input_ids']]
                    embeds = torch.stack(
                        [
                            outputs.last_hidden_state[i, start: end, :].mean(dim=0)
                            for i, (start, end) in enumerate(seq_locs)
                        ]
                    )
                else:
                    embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], 'last')
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def set_prefix(self, prefix: str):
        self.prefix = prefix

    def set_suffix(self, suffix: str):
        self.suffix = suffix


def main(all_pairs):

    model = DenseEncoder()

    for (task_type, task_name) in all_pairs:
        task_def: str = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
        prompt: str = get_detailed_instruct(task_def, args.special_token)
        model.set_prompt(prompt=prompt)

        eg_file_path = f'{args.examples_dir}/{task_name}.csv'
        eg_paris = []
        if args.zero_shot:
            eg_paris = []
        else:
            df = pd.read_csv(eg_file_path)
            for i in range(len(df)):
                eg_paris.append((get_detailed_instruct(
                    task_def,
                    args.special_token
                ) + df[df.keys()[0]][i], df[df.keys()[1]][i]))
        if args.special_token:
            if len(eg_paris) > 0:
                prefix = '\n\n'.join(['\n<response>'.join(eg_paris[idx]) for idx in range(len(eg_paris))]) + '\n\n'
            else:
                prefix = ''
            suffix = '\n<response>'
        else:
            if len(eg_paris) > 0:
                prefix = '\n\n'.join(['\nResponse: '.join(eg_paris[idx]) for idx in range(len(eg_paris))]) + '\n\n'
            else:
                prefix = ''
            suffix = '\nResponse:'
        model.set_prefix(prefix)
        model.set_suffix(suffix)

        logger.info('Set prompt: {}'.format(prompt))

        # disable l2 normalize for classification tasks, as it achieves slightly better results
        if task_type == 'Classification':
            logger.info('Set l2_normalize to False for classification task')
            model.l2_normalize = False
        else:
            model.l2_normalize = True
            logger.info('Set l2_normalize to {}'.format(model.l2_normalize))

        sub_eval = MTEB(tasks=[task_name], task_langs=['en'])

        logger.info('Running evaluation for task: {}, type: {}'.format(task_name, task_type))

        # eval_splits = ["test"] if "test" in task_cls.description["eval_splits"] else task_cls.description["eval_splits"]

        result_flag = False
        model.batch_size = args.batch_size
        while result_flag is False:
            try:
                sub_eval.run(
                    model,
                    output_folder=args.output_dir
                )
                result_flag = True
            except Exception as e:
                model.batch_size -= 4
                print(e)
        # sub_eval.run(
        #             model,
        #             output_folder=args.output_dir
        #         )


if __name__ == '__main__':
    processes = []
    multiprocessing.set_start_method('spawn')

    random.seed(30)
    args.task_types = [t for t in args.task_types if t.strip()]
    all_pairs = []
    for task_type in args.task_types:
        if task_type in tasks_desc.keys():
            for task_name in tasks_desc[task_type]:
                all_pairs.append((task_type, task_name))
    for task_type in tasks_desc.keys():
        for v in tasks_desc[task_type]:
            if v in args.task_types:
                all_pairs.append((task_type, v))
    all_pairs = list(set(all_pairs))

    main(all_pairs)