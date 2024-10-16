import collections
import re
import string
import copy
import logging
import numpy as np
from sklearn.metrics import f1_score
from typing import List, Dict
from rouge import Rouge
from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _normalize_answer(text, punc_chars, punc_repl):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)
    
    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)
    
    def white_space_fix(s):
        return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)
    return text


def normalize_squad(answer):
    """Normalization used in official SQuAD evaluation script."""
    return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def _metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
    """Computes the maximum of the metric over all ground truths."""
    return max(
        metric_fn(ground_truth, prediction) for ground_truth in ground_truths
    )


def _exact_match_score(target, prediction):
    return target == prediction


def _f1_score(target, prediction):
    """Computes token f1 score for a single target and prediction."""
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_metrics(targets, predictions, return_list=False):
    """Computes exact match and f1 QA scores, expecting pre-normalized text."""
    if len(targets) != len(predictions):
        raise ValueError("Number of targets and predictions must match.")
    if return_list:
        em=[
            _metric_max_over_ground_truths(_exact_match_score, t, p)
            for p, t in zip(predictions, targets)
        ]
        f1=[
            _metric_max_over_ground_truths(_f1_score, t, p)
            for p, t in zip(predictions, targets)
        ]
        return em, f1
    em = np.mean([
        _metric_max_over_ground_truths(_exact_match_score, t, p)
        for p, t in zip(predictions, targets)
    ])
    f1 = np.mean([
        _metric_max_over_ground_truths(_f1_score, t, p)
        for p, t in zip(predictions, targets)
    ])
    # em *= 100
    # f1 *= 100
    logger.info("EM = %.2f, F1 = %.2f", em, f1)
    #return {"em": em, "f1": f1}
    return em, f1


class App:
    def __init__(self):
        self.functions = {}

    def add(self, key):
        def adder(func):
            self.functions[key] = func
            return func

        return adder

    def __getitem__(self, __name: str):
        return self.functions[__name]


metric_dict = App()


@metric_dict.add("rouge")
def rouge(preds, labels, return_list=False):
    # https://github.com/pltrdy/rouge
    r1s, r2s, rls = [], [], []
    r = Rouge()
    for i in range(len(labels)):
        if "\n" not in preds[i]:
            preds[i] += "\n"  # to ensure rouge metrics
        if "\n" not in labels[i]:
            labels[i] += "\n"
        scores = r.get_scores(preds[i], labels[i])[0]
        r1s.append(scores["rouge-1"]["f"])
        r2s.append(scores["rouge-2"]["f"])
        rls.append(scores["rouge-l"]["f"])
    if return_list:  # used for scoring data
        return r1s
    r1 = sum(r1s) / len(r1s)
    r2 = sum(r2s) / len(r2s)
    rl = sum(rls) / len(rls)
    return r1, r2, rl


@metric_dict.add("squad")
def squad(labels, preds, return_list=False):
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and predictions
    """
    labels = [[normalize_squad(t) for t in u] for u in labels]
    preds = [normalize_squad(p) for p in preds]
    if return_list:  # used for scoring data
        em, f1 = qa_metrics(labels, preds, return_list=True)
        return f1
    em, f1 = qa_metrics(labels, preds)  # em,f1
    return em, f1



@metric_dict.add("simple_accuracy")
def simple_accuracy(preds, labels, return_list=False):
    if isinstance(preds[0], str):
        labels = [label.strip() for label in labels]
        preds = [pred.strip() for pred in preds]
    res = [int(preds[i] == labels[i]) for i in range(len(preds))]
    if return_list:
        return res
    acc = sum(res) / len(res)
    return acc


def compute_metrics(metric, labels, preds):
    assert len(preds) == len(labels)
    if metric == "acc":
        return {"acc": simple_accuracy(preds, labels)}
    elif metric == "rl":
        r1, r2, rl = rouge(preds, labels)
        # return {"r1": r1, "r2": r2, "rl": rl}
        return {"rl": rl}
    elif metric == "f1":
        f1 = f1_score(y_true=labels, y_pred=preds, pos_label='1')
        return {"f1": f1}
    elif metric == "em":
        em, f1 = squad(labels=labels, preds=preds)
        # return {"em": em, "f1": f1}
        return {"em": em}

def compute_scores(metric, preds, labels):
    if not isinstance(preds[0], str):
        preds = np.array(preds)
        labels = np.array(labels)
    scores = compute_metrics(metric, labels=labels, preds=preds)
    return scores

def flat_options(data):
    flat_data = []
    for e in data:
        for option in e['options']:
            flat_data.append({"query":e['query'], "few_shot":e['few_shot'], 'input_answer':option})
    return flat_data

def perplexity_to_choice(data, perplexity):
    inx = 0
    results = []
    for e in data:
        cur_perplexity = []
        for _ in e['options']:
            cur_perplexity.append(perplexity[inx])
            inx += 1
        ans = np.argmin(cur_perplexity)
        results.append(str(ans))
    return results


def get_length(tokenizer, text):
    tokenized_example = tokenizer.encode_plus(text,truncation=False, return_tensors='pt')
    shape = tokenized_example.input_ids.squeeze().shape
    if len(shape)==0:
        return 1
    else:
        return int(shape[0])


def get_prompt_length(tokenizer, prompts_list, question, n_tokens_in_prompt: int=1024):
    lengths_list = [get_length(tokenizer, prompt) for prompt in prompts_list]
    q_length = get_length(tokenizer, question)
    max_prompts = np.searchsorted(np.cumsum(lengths_list), n_tokens_in_prompt - q_length)
    return max_prompts


def _llm_generation_func(examples: Dict[str, List],
                    tokenizer: PreTrainedTokenizer,
                    example_num: int=8,
                    max_input_tokens: int=1024,
                    add_llama_inst: bool=False):
    texts = []
    n_tokens_in_prompt = max_input_tokens
    if add_llama_inst:
        n_tokens_in_prompt -= 8

    for i in range(len(examples['query'])):
        prompts_list = examples['few_shot'][i][::-1]
        max_prompts = get_prompt_length(
            tokenizer=tokenizer, 
            prompts_list=prompts_list, 
            question=examples['query'][i], 
            n_tokens_in_prompt=n_tokens_in_prompt
        )
        example_num = min(example_num, max_prompts)
        
        inputs = prompts_list[:example_num]
        
        inputs.append(examples['query'][i])
        
        if add_llama_inst:
            inputs = "[INST] " + "\n".join(inputs) + " [/INST]"
        else:
            inputs = "\n".join(inputs)+'\n'

        texts.append(inputs)
    return tokenizer(texts, return_tensors="pt", padding=True, max_length=1024,  return_token_type_ids=False)


def _llm_perplexity_func(examples: Dict[str, List],
                    tokenizer: PreTrainedTokenizer,
                    example_num: int=8,
                    max_input_tokens: int=1024,
                    add_llama_inst: bool=False):
    texts = []
    answers = []
    n_tokens_in_prompt = max_input_tokens
    if add_llama_inst:
        n_tokens_in_prompt -= 8

    for i in range(len(examples['query'])):
        prompts_list = examples['few_shot'][i][::-1]
        max_prompts = get_prompt_length(tokenizer=tokenizer, 
                                        prompts_list=prompts_list, 
                                        question=examples['query'][i], 
                                        n_tokens_in_prompt=n_tokens_in_prompt)
        example_num = min(example_num, max_prompts)
        
        inputs = prompts_list[:example_num]
        
        inputs.append(examples['query'][i])
        if add_llama_inst:
            # NOTE: two more spaces after [/INST] 
            inputs = "[INST] " + "\n".join(inputs) + " [/INST]  " + examples['input_answer'][i].lstrip()
        else:
            inputs = "\n".join(inputs)+'\n ' + examples['input_answer'][i] # add a space after \n to split input and answer

        texts.append(inputs)
        answers.append(examples['input_answer'][i])
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False)
    
    labels = copy.deepcopy(inputs['input_ids'])
    for i, ans in enumerate(answers):
        ans_ids = tokenizer.encode(ans, add_special_tokens=False)
        labels[i][:-len(ans_ids)] = -100

    inputs['labels'] = labels
    return inputs
