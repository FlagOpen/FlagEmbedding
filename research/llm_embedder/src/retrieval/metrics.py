import os
import json
import logging
import inspect
import numpy as np
from tqdm import tqdm
from .evalnq import evaluate_nq
from ..utils.util import makedirs, split_file_dir_name_ext

logger = logging.getLogger(__name__)


class RetrievalMetric:
    """Class for computing metrics and some post-processings."""
    @classmethod
    def get_metric_fn(cls, metric_names, **kwds):
        assert isinstance(metric_names, list) or isinstance(metric_names, tuple), "You must pass metric_names in a list or tuple!"
        all_metrics = {}
        # get all methods
        all_implemented_fns = [x[0] for x in inspect.getmembers(cls, predicate=inspect.isfunction) if not x[0].startswith("_")]

        def compute_metrics(*args, **kwargs):
            for metric_name in metric_names:
                # call corresponding method
                if metric_name in all_implemented_fns:
                    metric_fn = getattr(cls, metric_name)
                    metric = metric_fn(**kwds)(*args, **kwargs)
                    # NOTE: some metric_fn are only used for post-processing and saving results, which return None by default
                    if metric is not None:
                        all_metrics.update(metric)
                else:
                    raise NotImplementedError(f"Metric {metric_name} not implemented!")
            return all_metrics
        return compute_metrics
    
    @staticmethod
    def _get_save_path(eval_data, output_dir=None, field="result", save_name=None):
        """
        if output_dir is None:
            -> {eval_data_dir}/{eval_data_name}.{field}.{save_name}.{eval_data_ext}
        else:
            -> {output_dir}/{eval_data_name}.{field}.{save_name}.{eval_data_ext}
        """
        eval_data_dir, eval_data_name, eval_data_ext = split_file_dir_name_ext(eval_data)
        if output_dir is None:
            output_dir = eval_data_dir
        fields = [eval_data_name, field]
        if save_name is not None:
            fields.append(save_name)
        save_path = os.path.join(output_dir, ".".join(fields) + eval_data_ext)
        makedirs(save_path)
        return save_path

    @staticmethod
    def _save_result(query_ids, preds, result_path, scores=None):
        if query_ids is None and preds is None:
            logger.warning("No query_ids and preds provided for _save_result, skipping!")
            return

        with open(result_path, "w") as f:
            for i, (query_id, pred) in enumerate(zip(query_ids, preds)):
                res = {
                    "query_id": query_id,
                    "pred": pred,
                }
                if scores is not None:
                    res["score"] = scores[i]
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    @staticmethod
    def _load_result(result_path):
        logger.info(f"loading retrieval results from {result_path}...")
        all_query_ids = []
        all_preds = []
        all_scores = None
        with open(result_path) as f:
            for line in f:
                item = json.loads(line.strip())
                all_query_ids.append(item["query_id"])
                all_preds.append(item["pred"])
                if "scores" in item:
                    if all_scores is None:
                        all_scores = []
                    all_scores.append(item["scores"])

        if all_scores is not None:
            return all_query_ids, all_preds, all_scores
        else:
            return all_query_ids, all_preds

    @staticmethod
    def _clean_pred(pred, score=None):
        if isinstance(pred, np.ndarray):
            valid_pos = pred > -1
            pred = pred[valid_pos].tolist()
            if score is not None:
                score = score[valid_pos].tolist()
        else:
            valid_pos = [i for i, x in enumerate(pred) if x > -1]
            pred = [pred[i] for i in valid_pos]
            if score is not None:
                score = [score[i] for i in valid_pos]
        if score is not None:
            return pred, score
        else:
            return pred
    
    @staticmethod
    def _prepare_label(eval_data):
        labels = {}
        with open(eval_data) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                if "query_id" in item:
                    query_id = item["query_id"]
                else:
                    query_id = i
                # get the indices of the positives w.r.t. the corpus
                label = item.get("pos_index", None)
                labels[query_id] = label
        return labels

    @staticmethod
    def mrr(eval_data=None, cutoffs=[10], **kwds):
        metric_name = inspect.currentframe().f_code.co_name
        if eval_data is not None:
            data_labels = RetrievalMetric._prepare_label(eval_data)

        def compute_metric(query_ids, preds, labels=None, **kwargs):
            if labels is None:
                labels = data_labels
            
            if len(preds) != len(labels):
                logger.warning(f"There are {len(preds)} queries in predictions while {len(labels)} queries in labels!")
            
            mrrs = np.zeros(len(cutoffs))
            for query_id, pred in zip(query_ids, preds):
                label = labels[query_id]
                pred = RetrievalMetric._clean_pred(pred)

                jump = False
                for i, x in enumerate(pred, 1):
                    if x == -1:
                        break
                    if x in label:
                        for k, cutoff in enumerate(cutoffs):
                            if i <= cutoff:
                                mrrs[k] += 1 / i
                        jump = True
                    if jump:
                        break
            mrrs /= len(preds)

            metric = {}
            for i, cutoff in enumerate(cutoffs):
                mrr = mrrs[i]
                metric[f"{metric_name}@{cutoff}"] = mrr

            return metric
        return compute_metric

    @staticmethod
    def recall(eval_data=None, cutoffs=[10], **kwds):
        metric_name = inspect.currentframe().f_code.co_name
        if eval_data is not None:
            data_labels = RetrievalMetric._prepare_label(eval_data)

        def compute_metric(query_ids, preds, labels=None, **kwargs):
            if labels is None:
                labels = data_labels

            if len(preds) != len(labels):
                logger.warning(f"There are {len(preds)} queries in predictions while {len(labels)} queries in labels!")

            recalls = np.zeros(len(cutoffs))
            for query_id, pred in zip(query_ids, preds):
                label = labels[query_id]
                pred = RetrievalMetric._clean_pred(pred)
                for k, cutoff in enumerate(cutoffs):
                    recall = np.intersect1d(label, pred[:cutoff])
                    recalls[k] += len(recall) / len(label)

            recalls /= len(preds)

            metric = {}
            for i, cutoff in enumerate(cutoffs):
                recall = recalls[i]
                metric[f"{metric_name}@{cutoff}"] = recall

            return metric
        return compute_metric
    
    @staticmethod
    def ndcg(eval_data=None, cutoffs=[10], **kwds):
        metric_name = inspect.currentframe().f_code.co_name
        if eval_data is not None:
            data_labels = RetrievalMetric._prepare_label(eval_data)

        def compute_metric(query_ids, preds, labels=None, **kwargs):
            if labels is None:
                labels = data_labels

            if len(preds) != len(labels):
                logger.warning(f"There are {len(preds)} queries in predictions while {len(labels)} queries in labels!")
            
            ndcgs = np.zeros(len(cutoffs))
            for query_id, pred in zip(query_ids, preds):
                label = labels[query_id]

                pred = RetrievalMetric._clean_pred(pred)
                ndcg = np.zeros(len(cutoffs))
                idcg = np.zeros(len(cutoffs))

                for i, x in enumerate(pred, 1):
                    if x in label:
                        for k, cutoff in enumerate(cutoffs):
                            if i <= cutoff:
                                ndcg[k] += 1 / np.log2(i + 1)
                for j, y in enumerate(label, 1):
                    for k, cutoff in enumerate(cutoffs):
                        if j <= cutoff:
                            idcg[k] += 1 / np.log2(j + 1)
                ndcgs += ndcg / idcg
            ndcgs /= len(preds)

            metric = {}
            for i, cutoff in enumerate(cutoffs):
                ndcg = ndcgs[i]
                metric[f"{metric_name}@{cutoff}"] = ndcg
            return metric
        return compute_metric

    @staticmethod
    def nq(eval_data, corpus, cache_dir=None, **kwds):
        def compute_metric(query_ids, preds, **kwargs):
            # collect retrieval result
            retrieval_result = {}
            for i, pred in enumerate(preds):
                retrieval_result[i] = RetrievalMetric._clean_pred(pred)

            metrics = evaluate_nq(retrieval_result, eval_data=eval_data, corpus=corpus, cache_dir=cache_dir)
            return metrics
        return compute_metric

    @staticmethod
    def collate_key(eval_data, save_name, corpus, output_dir=None, save_to_output=False, **kwds):
        """
        Collate retrieval results for evaluation. 
        Append a 'keys' column in the eval_data where each key is a piece of retrieved text;
        Delete 'pos' and 'neg' column.
        If output_dir is None, save at {eval_data}.keys.{save_name}.json
        Else, save at {output_dir}.keys.{save_name}.json
        """
        def collate(query_ids, preds, **kwargs):
            query_id_2_pred = {}
            for query_id, pred in zip(query_ids, preds):
                pred = RetrievalMetric._clean_pred(pred)
                query_id_2_pred[query_id] = pred
            del query_ids
            del preds

            if save_to_output and output_dir is not None:
                save_path = RetrievalMetric._get_save_path(eval_data, output_dir, field="key", save_name=save_name)
            else:
                save_path = RetrievalMetric._get_save_path(eval_data, None, field="key", save_name=save_name)

            logger.info(f"saving key to {save_path}...")
            with open(eval_data) as f, open(save_path, "w") as g:
                for line in tqdm(f, desc="Collating key"):
                    item = json.loads(line)
                    query_id = item["query_id"]
                    # NOTE: some queries may not correspond to any keys (especially in case of BM25), just skip them
                    if query_id not in query_id_2_pred:
                        item["key"] = []
                        item["key_index"] = []
                    else:
                        pred = query_id_2_pred[query_id]
                        item["key"] = corpus[pred]["content"]
                        item["key_index"] = pred

                    # delete pos, neg, and teacher scores because they do not comply with new keys
                    # if "pos" in item:
                    #     del item["pos"]
                    # if "neg" in item:
                    #     del item["neg"]
                    # if "pos_index" in item:
                    #     del item["pos_index"]
                    # if "neg_index" in item:
                    #     del item["neg_index"]
                    # if "teacher_scores" in item:
                    #     del item["teacher_scores"]
                    g.write(json.dumps(item, ensure_ascii=False) + "\n")
        return collate

    @staticmethod
    def collate_neg(eval_data, save_name, corpus, max_neg_num=100, filter_answers=False, output_dir=None, save_to_output=False, **kwds):
        """
        Collate retrieval results for training. 
        Append 'pos' and 'neg' columns in the eval_data where each element is a piece of retrieved text;
        Save at {output_dir}.neg.{save_name}.json
        """
        def collate(query_ids, preds, **kwargs):
            query_id_2_pred = {}
            for query_id, pred in zip(query_ids, preds):
                pred = RetrievalMetric._clean_pred(pred)
                query_id_2_pred[query_id] = pred
            del query_ids
            del preds

            if save_to_output and output_dir is not None:
                save_path = RetrievalMetric._get_save_path(eval_data, output_dir, field="neg", save_name=save_name)
            else:
                save_path = RetrievalMetric._get_save_path(eval_data, None, field="neg", save_name=save_name)

            logger.info(f"saving {max_neg_num} negatives to {save_path}...")
            with open(eval_data) as f, open(save_path, "w") as g:
                for line in tqdm(f, desc="Collating Negatives"):
                    item = json.loads(line)
                    query_id = item["query_id"]

                    # NOTE: some queries may not correspond to any negatives (especially in case of BM25), just skip them
                    if query_id not in query_id_2_pred:
                        continue

                    pred = query_id_2_pred[query_id]

                    if "pos" in item:
                        pos = set(item["pos"])
                    else:
                        # sometime we do not have pre-defined pos, instead, the pos will be selected from neg based on teacher scores
                        pos = []

                    # first filter out positive documents
                    if "pos_index" in item:
                        pos_index = item["pos_index"]
                        pred = [i for i in pred if i != pos_index]

                    neg = corpus[pred]["content"]

                    # remove key that is the same as pos
                    # NOTE: here we do not use pos_index to distinguish pos and neg, because different pos_index may correpond to the same content due to duplication in the corpus
                    if filter_answers:
                        answers = item.get("answers", [])
                        valid_index = [i for i, x in enumerate(neg) if (x not in pos) and (not any(a.lower() in x.lower() for a in answers))]
                    else:
                        valid_index = [i for i, x in enumerate(neg) if x not in pos]
                    valid_index = valid_index[:max_neg_num]

                    neg = [neg[i] for i in valid_index]
                    neg_index = [pred[i] for i in valid_index]

                    item["neg"] = neg
                    item["neg_index"] = neg_index

                    # remove teacher scores because they are for previous pos and neg
                    if "teacher_scores" in item:
                        del item["teacher_scores"]

                    g.write(json.dumps(item, ensure_ascii=False) + "\n")
        return collate
    
    @staticmethod
    def collate_score(eval_data, save_name, output_dir=None, save_to_output=False, **kwds):
        """
        Collate scores generated by the reranking model. 
        Append 'teacher_scores' column in the eval_data where each element is the score of 'pos' unioned 'neg';
        If output_dir is None, save at {eval_data}.score.{save_name}.json
        Else, save at {output_dir}.score.{save_name}.json
        """
        def collate(query_ids, preds, scores, **kwargs):
            query_id_2_pred = {}
            for query_id, pred, score in zip(query_ids, preds, scores):
                pred, score = RetrievalMetric._clean_pred(pred, score)
                query_id_2_pred[query_id] = (pred, score)
            del query_ids
            del preds
            del scores
            
            if save_to_output and output_dir is not None:
                save_path = RetrievalMetric._get_save_path(eval_data, output_dir, field="scored", save_name=save_name)
            else:
                save_path = RetrievalMetric._get_save_path(eval_data, None, field="scored", save_name=save_name)

            logger.info(f"saving scores to {save_path}...")
            with open(eval_data) as f, open(save_path, "w") as g:
                for line in tqdm(f, desc="Collating Scores"):
                    item = json.loads(line)
                    query_id = item["query_id"]

                    pred, score = query_id_2_pred[query_id]
                    
                    # NOTE: there must be key_index
                    if "pos_index" in item:
                        key_index = item["pos_index"] + item["neg_index"]
                    elif "key_index" in item:
                        key_index = item["key_index"]
                    else:
                        key_index = list(range(len(pred)))

                    key_index_2_score = {k: s for k, s in zip(pred, score)}
                    teacher_scores = [key_index_2_score[ki] for ki in key_index]
                    item["teacher_scores"] = teacher_scores

                    g.write(json.dumps(item, ensure_ascii=False) + "\n")
        return collate
