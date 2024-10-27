# Ref: https://github.com/facebookresearch/contriever
import regex
import unicodedata
from functools import partial
from typing import List


class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []
    for i, text in enumerate(ctxs):
        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue
        hits.append(has_answer(answers, text, tokenizer))
    return hits


def evaluate_recall_qa(ctxs, answers, k=100):
    # compute Recall@k for QA task
    data = []
    assert len(ctxs) == len(answers)
    for i in range(len(ctxs)):
        _ctxs, _answers = ctxs[i], answers[i]
        data.append({
            'answers': _answers,
            'ctxs': _ctxs,
        })
    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer)
    
    scores = map(get_score_partial, data)
    
    n_docs = len(data[0]['ctxs'])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    k = min(k, len(top_k_hits))
    return top_k_hits[k - 1] / len(data)
