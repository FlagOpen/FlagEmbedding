import os
import json
import inspect
import numpy as np
from rouge import Rouge
from tqdm import tqdm
from transformers.utils import logging
from .util import makedirs, split_file_dir_name_ext, normalize_text

logger = logging.get_logger(__name__)


class Metric:
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
    def _save_result(indices, preds, path, data=None):
        if data is not None:
            items = {}
            with open(data, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    item = json.loads(line)
                    items[i] = item

        with open(path, "w") as f:
            for i, (index, pred) in enumerate(zip(indices, preds)):
                if data is not None:
                    item = items[index]
                else:
                    item = {}
                
                item["index"] = index
                item["pred"] = pred
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_result(path):
        logger.info(f"loading retrieval results from {path}...")
        results = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                query_id = item["query_id"]
                results[query_id] = item
        return results
    
    @staticmethod
    def _prepare_label(eval_data):
        labels = {}
        with open(eval_data) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                # get the indices of the positives w.r.t. the corpus
                label = item["output"]
                labels[i] = label
        return labels
    
    @staticmethod
    def rouge(eval_data=None, **kwds):
        if eval_data is not None:
            data_labels = Metric._prepare_label(eval_data)
        
        rouge = Rouge()

        def compute_metric(indices, preds, labels=None, **kwargs):
            if labels is None:
                labels = data_labels

            if len(preds) != len(labels):
                logger.warning(f"There are {len(preds)} queries in predictions while {len(labels)} queries in labels!")

            labels = [labels[query_id] for query_id in indices]

            preds = normalize_text(preds)
            labels = normalize_text(labels)

            # filter empty preditions
            preds = [":)" if len(pred) == 0 else pred for pred in preds]

            score = rouge.get_scores(preds, labels, avg=True)

            metric = {
                "rouge-1": score["rouge-1"]["f"],
                "rouge-2": score["rouge-2"]["f"],
                "rouge-l": score["rouge-2"]["f"],
            }
            return metric
        return compute_metric
    
    @staticmethod
    def acc(eval_data=None, **kwds):
        if eval_data is not None:
            data_labels = Metric._prepare_label(eval_data)

        def compute_metric(indices, preds, labels=None, **kwargs):
            if labels is None:
                labels = data_labels

            if len(preds) != len(labels):
                logger.warning(f"There are {len(preds)} queries in predictions while {len(labels)} queries in labels!")

            labels = [labels[query_id] for query_id in indices]

            preds = normalize_text(preds)
            labels = normalize_text(labels)

            overlap = 0
            for pred, label in zip(preds, labels):
                if pred == label:
                    overlap += 1

            metric = {
                "acc": overlap / len(preds),
            }
            return metric
        return compute_metric
