import os.path
import random
import sys
import time
from dataclasses import dataclass
import re

import datasets
import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer

from arguments import DataArguments
from nltk.corpus import stopwords

class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            args: DataArguments,
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train',
                                                 cache_dir=args.cache_path)

        self.args = args
        self.total_len = len(self.dataset)

        self.remove_stop_words = args.remove_stop_words
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['!', ',' ,'.' ,'?'])

        self.max_length = args.cutoff_len
        self.tokenizer = tokenizer

        self.prefix = '"'
        self.suffix = ['", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>',
                       '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>']
        self.prefix_ids = self.tokenizer(self.prefix, truncation=True, max_length=self.max_length, return_tensors=None)['input_ids']
        self.suffix_ids = self.tokenizer(self.suffix, truncation=True, max_length=self.max_length, return_tensors=None, add_special_tokens=False)['input_ids']

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        prefix = self.prefix
        prefix_ids = self.prefix_ids

        inp = self.dataset[item]['input']

        suffix_ids_summarize = self.suffix_ids[0]
        suffix_ids_predict = self.suffix_ids[1]

        oup_summarize = self.dataset[item]['output_summarize']
        oup_predict = self.dataset[item]['output_predict']

        input_ids = self.tokenizer(inp,
                                   truncation=True,
                                   max_length=self.max_length - len(prefix_ids) - len(suffix_ids_summarize) - len(suffix_ids_predict),
                                   padding=False,
                                   return_tensors=None,
                                   add_special_tokens=False)
        result = dict()
        result['input_ids'] = prefix_ids + input_ids['input_ids'] + suffix_ids_summarize + suffix_ids_predict
        result['attention_mask'] = [1] * len(result['input_ids'])
        result['labels'] = [-100] * len(prefix_ids) + result['input_ids'][
            len(prefix_ids) : len(result['input_ids']) - len(suffix_ids_summarize) - len(suffix_ids_predict)] + [
            -100] * (len(suffix_ids_summarize) + len(suffix_ids_predict))
        if self.remove_stop_words:
            # oup = re.sub(r'[\d\W_]+', ' ', oup)
            oup_summarize = re.sub(r'[\W_]+', ' ', oup_summarize)
            oup_summarize = ' '.join([word for word in oup_summarize.split() if word.lower() not in self.stop_words])

            oup_predict = re.sub(r'[\W_]+', ' ', oup_predict)
            oup_predict = ' '.join([word for word in oup_predict.split() if word.lower() not in self.stop_words])
        return result, oup_summarize, oup_predict


@dataclass
class EmbedCollator(DataCollatorForSeq2Seq):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    cutoff_len: int = 512

    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        inputs = []
        outputs_summarize = []
        outputs_predict = []
        for e in features:
            inputs.append(e[0])
            outputs_summarize.append(e[1])
            outputs_predict.append(e[2])

        labels = [feature["labels"] for feature in inputs] if "labels" in inputs[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in inputs:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        inputs = self.tokenizer.pad(
            inputs,
            padding=self.padding,
            max_length=self.cutoff_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=inputs["labels"])
            inputs["decoder_input_ids"] = decoder_input_ids

        outputs_summarize_collated = self.tokenizer(
            outputs_summarize,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )

        outputs_predict_collated = self.tokenizer(
            outputs_predict,
            padding=True,
            truncation=True,
            max_length=self.cutoff_len,
            return_tensors="pt",
        )

        return {"input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "labels": inputs['labels'],
                "output_summarize_ids": outputs_summarize_collated['input_ids'],
                "output_predict_ids": outputs_predict_collated['input_ids']}