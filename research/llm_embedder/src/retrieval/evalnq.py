import os
import datasets
import regex
import unicodedata
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm



class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncase=False):
        tokens = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()
            # Format data
            if uncase:
                tokens.append(token.lower())
            else:
                tokens.append(token)
        return tokens


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string.
    """
    text = _normalize(text)

    # Answer is a list of possible strings
    text = tokenizer.tokenize(text, uncase=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncase=True)

        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


class EvalDataset(Dataset):
    def __init__(self, retrieval_result, eval_dataset, corpus):
        self.corpus = corpus
        self.eval_dataset = eval_dataset
        self.retrieval_result = retrieval_result
        self.tokenizer = SimpleTokenizer()

    def __getitem__(self, qidx):
        res = self.retrieval_result[qidx]
        hits = []
        for i, tidx in enumerate(res):
            if tidx == -1:
                hits.append(False)
            else:
                hits.append(has_answer(self.eval_dataset[qidx]["answers"], self.corpus[tidx]["content"], self.tokenizer))
        return hits

    def __len__(self):
        return len(self.retrieval_result)


def evaluate_nq(retrieval_result: dict, eval_data: datasets.Dataset, corpus: datasets.Dataset, num_workers=16, batch_size=16, cache_dir=None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if isinstance(eval_data, str):
        eval_dataset = datasets.load_dataset("json", data_files=eval_data, split="train", cache_dir=cache_dir)
    elif isinstance(eval_data, datasets.Dataset):
        eval_dataset = eval_data
    else:
        raise ValueError(f"Expected eval_data of type str/Dataset, found {type(eval_data)}!")

    if isinstance(corpus, str):
        corpus = datasets.load_dataset("json", data_files=corpus, split="train", cache_dir=cache_dir)
    elif isinstance(corpus, datasets.Dataset):
        pass
    else:
        raise ValueError(f"Expected corpus of type str/Dataset, found {type(corpus)}!")

    dataset = EvalDataset(retrieval_result, eval_dataset=eval_dataset, corpus=corpus)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)

    final_scores = []
    for scores in tqdm(dataloader, total=len(dataloader), ncols=100, desc="Computing Metrics"):
        final_scores.extend(scores)

    relaxed_hits = np.zeros(max(*[len(x) for x in retrieval_result.values()], 100))
    for question_hits in final_scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            relaxed_hits[best_hit:] += 1

    relaxed_recall = relaxed_hits / len(retrieval_result)

    return {
        "recall@1": round(relaxed_recall[0], 4),
        "recall@5": round(relaxed_recall[4], 4),
        "recall@10": round(relaxed_recall[9], 4),
        "recall@20": round(relaxed_recall[19], 4),
        "recall@100": round(relaxed_recall[99], 4)
    }
