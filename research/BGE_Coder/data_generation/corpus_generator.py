import os
import random
import datasets
from tqdm import tqdm
from typing import List, Tuple

from utils import clean_code
from constant import DocLength


class CorpusGenerator:
    def __init__(
        self,
        cache_dir: str = None,
    ):
        self.cache_dir = cache_dir

    def _load_corpus(self, corpus_dir: str, doc_length: List[str], external_path: List[str],
                     source_language: str, stop_threshold: int = -1):
        """
        Load availavle documents for a given task from the CoIR-Retrieval dataset.
        """

        corpus_list = []

        if corpus_dir is not None and os.path.exists(corpus_dir):
            file_list = os.listdir(corpus_dir)
            random.shuffle(file_list)
            
            for file in file_list:
                flag = False
                if not file.endswith('.jsonl'):
                    flag = False
                for d_length in doc_length:
                    d_length = DocLength[d_length].value
                    if d_length in file:
                        flag = True
                if flag is False:
                    continue
                file_path = os.path.join(corpus_dir, file)
                corpus = datasets.load_dataset('json', data_files=file_path, cache_dir=self.cache_dir)['train']
                for data in tqdm(corpus, desc="Loading corpus"):
                    if source_language is None:
                        lang = os.path.basename(corpus_dir)
                        data['language'] = lang
                    else:
                        data['language'] = source_language
                    
                    text = clean_code(data["text"], data["language"], length_threshold=200)
                    data["text"] = text
                    if text != '':
                        corpus_list.append(data)
                
                if stop_threshold > 0 and len(corpus_list) > stop_threshold:
                    break
                break

        for ep in external_path:
            if os.path.exists(ep):
                corpus = datasets.load_dataset('json', data_files=ep, cache_dir=self.cache_dir)['train']
                for data in tqdm(corpus, desc="Loading corpus"):
                    if source_language is None:
                        lang = os.path.basename(os.path.dirname(ep))
                        data['language'] = lang
                    else:
                        data['language'] = source_language
                    
                    # useful when the text is not present in the data
                    if "text" not in data:
                        data["text"] = data["pos"][0]
                    
                    corpus_list.append(data)
                    text = clean_code(data["text"], lang, length_threshold=200)
                    data["text"] = text
                    if text != '':
                        corpus_list.append(data)

        return corpus_list

    def run(
        self,
        num_samples: int = -1,
        max_corpus: int = -1,
        corpus_dir: str = None,
        doc_length: List[str] = ["len_0_500"],
        external_path: List[str] = None,
        source_language: str = None
    ) -> Tuple[List[dict], List[dict]]:
        stop_threshold = max(num_samples * 10, max_corpus * 2)
        corpus_list = self._load_corpus(
            corpus_dir, doc_length, external_path, source_language, stop_threshold
        )

        if num_samples > 0 and num_samples < len(corpus_list):
            small_corpus_list = random.sample(corpus_list, num_samples)
        else:
            small_corpus_list = corpus_list
        
        if max_corpus > 0 and max_corpus < len(corpus_list):
            corpus_list = random.sample(corpus_list, max_corpus)
        else:
            corpus_list = corpus_list

        return small_corpus_list, corpus_list
