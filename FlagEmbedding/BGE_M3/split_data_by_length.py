"""
python split_data_by_length.py \
--input_path train_data \
--output_dir train_data_split \
--cache_dir .cache \
--log_name .split_log \
--length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
--model_name_or_path BAAI/bge-m3 \
--num_proc 16 \
--overwrite False
"""
import os
import json
import math
import time
import argparse
import datasets
from tqdm import tqdm
from pprint import pprint
from transformers import AutoTokenizer
from datasets import load_dataset, Features, Value, Sequence


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='the path of input datas')
    parser.add_argument('--output_dir', type=str, required=True, help='the dir of output datas')
    parser.add_argument('--cache_dir', type=str, default=None, help='the cache dir')
    parser.add_argument('--log_name', type=str, default='.split_log', help='the name of log file, default: `.split_log`, which will be saved to `output_dir`')
    parser.add_argument('--length_list', type=int, default=[0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000], nargs='+', help='the length list to split')
    parser.add_argument('--model_name_or_path', type=str, default='BAAI/bge-m3', help='the model name or path of the tokenizer')
    parser.add_argument('--num_proc', type=int, default=16, help='the number of process, default: 16')
    parser.add_argument('--overwrite', action='store_true', default=False, help='whether to overwrite the output file, default: False')
    args = parser.parse_args()
    return args


class SplitByLengthHandler:
    def __init__(self,
                 model_name_or_path: str,
                 cache_dir: str=None,
                 num_proc: int=16,
                 length_list: list=[0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000],
                 overwrite: bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.length_ranges_list = self._get_length_ranges_list(length_list)
        self.overwrite = overwrite

        pprint(self.length_ranges_list)

        def _map_func(examples):
            results = {}
            results['idx'] = []
            results['max_length'] = []
            for i in range(len(examples['query'])):
                idx = examples['idx'][i]
                query = examples['query'][i]
                pos, neg = examples['pos'][i], examples['neg'][i]
                all_texts = [query] + pos + neg

                max_len = 0
                for x in all_texts:
                    tokenized_x = self.tokenizer(x)['input_ids']
                    if len(tokenized_x) > max_len:
                        max_len = len(tokenized_x)
                
                results['idx'].append(idx)
                results['max_length'].append(max_len)
            return results

        self._map_func = _map_func

    @staticmethod
    def _get_length_ranges_list(length_list: list):
        length_ranges_list = []
        length_list = sorted(length_list)
        for i in range(len(length_list)):
            length_l = length_list[i]
            if i == len(length_list) - 1:
                length_r = math.inf
            else:
                length_r = length_list[i + 1]
            assert 0 <= length_l < length_r
            length_ranges_list.append((length_l, length_r))

        return length_ranges_list

    def _process_dir(self, dir_path: str, output_dir: str):
        assert os.path.isdir(dir_path)
        log_info_list = []
        for file in tqdm(os.listdir(dir_path), desc=f'processing {dir_path}'):
            file_path = os.path.join(dir_path, file)
            if not file_path.endswith('.jsonl'):
                print(f"skip {file_path} ...")
                continue

            output_path = os.path.join(output_dir, '.'.join(file.split('.')[:-1]))
            log_info = self._process_file(file_path, output_path)
            log_info_list.append(log_info)
        return log_info_list

    def _process_file(self, file_path: str, output_path: str):
        assert not os.path.isdir(file_path)

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        features = Features({
            'query': Value('string'),
            'pos': Sequence(Value('string')),
            'neg': Sequence(Value('string'))
        })
        kd_features = Features({
            'query': Value('string'),
            'pos': Sequence(Value('string')),
            'neg': Sequence(Value('string')),
            'pos_scores': Sequence(Value('float')),
            'neg_scores': Sequence(Value('float'))
        })
        try:
            dataset = load_dataset('json', data_files=file_path, cache_dir=self.cache_dir, features=features)['train']
        except:
            dataset = load_dataset('json', data_files=file_path, cache_dir=self.cache_dir, features=kd_features)['train']

        dataset_with_idx_list = []
        for i, data in enumerate(dataset):
            data['idx'] = i
            dataset_with_idx_list.append(data)
        dataset_with_idx = datasets.Dataset.from_list(dataset_with_idx_list)
        
        mapped_dataset = dataset_with_idx.map(self._map_func, batched=True, num_proc=self.num_proc)
        
        split_info_dict = {}
        for length_l, length_r in self.length_ranges_list:
            save_path = output_path + f'_len-{length_l}-{length_r}.jsonl'
            if os.path.exists(save_path) and not self.overwrite:
                print(f'{save_path} exists, skip')
                continue

            idxs = mapped_dataset.filter(lambda x: length_l <= x['max_length'] < length_r, num_proc=self.num_proc)
            split_dataset = dataset_with_idx.select(idxs['idx'])
            split_dataset = split_dataset.remove_columns('idx')

            split_info_dict[f'len-{length_l}-{length_r}'] = len(split_dataset)

            if len(split_dataset) > 0:
                split_dataset.to_json(save_path, force_ascii=False)

        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        size = len(dataset)
        avg_length = sum(mapped_dataset['max_length']) / size
        log_info = {
            'file_name': os.path.basename(file_path),
            'size': size,
            'avg_length': avg_length,
            'file_path': file_path,
            'start_time': start_time,
            'end_time': end_time,
            'split_info': split_info_dict
        }
        return log_info

    def run(self, input_path: str, output_dir: str, log_name: str=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if log_name is None:
            log_path = os.path.join(output_dir, '.split_log')
        else:
            log_path = os.path.join(output_dir, log_name)

        log_info_list = []

        if os.path.isdir(input_path):
            log_info_list = self._process_dir(input_path, output_dir)
        else:
            file_name = os.path.basename(input_path)
            output_path = os.path.join(output_dir, '.'.join(file_name.split('.')[:-1]))
            log_info = self._process_file(input_path, output_path)
            log_info_list.append(log_info)

        with open(log_path, 'a', encoding='utf-8') as f:
            for log_info in log_info_list:
                json.dump(log_info, f, ensure_ascii=False)
                f.write('\n')


if __name__ == '__main__':
    args = get_args()
    input_path = args.input_path
    output_dir = args.output_dir
    log_name = args.log_name

    handler = SplitByLengthHandler(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
        length_list=args.length_list if isinstance(args.length_list, list) else [args.length_list],
        overwrite=args.overwrite
    )

    handler.run(
        input_path=input_path,
        output_dir=output_dir,
        log_name=log_name
    )
    print('\nDONE!')
