import os
import json
import inspect
import numpy as np
from functools import partial
from rouge import Rouge
from tqdm import tqdm
from transformers.utils import logging
from .utils import makedirs, split_file_dir_name_ext, normalize_text

logger = logging.get_logger(__name__)


class Metric:
    """Class for computing metrics and some post-processings."""
    @classmethod
    def get_metric_fn(cls, metrics, **kwds):
        assert isinstance(metrics, list) or isinstance(metrics, tuple), "You must pass metric_names in a list or tuple!"
        return_metrics = {}
        # get all methods
        metric_fns = []

        all_metric_names = [x[0] for x in inspect.getmembers(cls, predicate=inspect.isfunction) if not x[0].startswith("get_")]
        for metric_name in metrics:
            if metric_name in all_metric_names:
                metric_fns.append(partial(getattr(cls, metric_name), **kwds))
            else:
                raise NotImplementedError(f"Metric {metric_name} not implemented!")

        def compute_metrics(*args, **kwargs):
            for metric_fn in metric_fns:
                # call corresponding method
                metric = metric_fn(*args, **kwargs)
                # NOTE: some metric_fn are only used for post-processing and saving results, which return None by default
                if metric is not None:
                    return_metrics.update(metric)
            return return_metrics
        return compute_metrics
    
    def get_save_path(eval_data, output_dir=None, field="result", save_name=None):
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

    def save_result(preds, labels, save_path, indices=None, **kwargs):
        if len(preds) != len(labels):
            logger.warning(f"There are {len(preds)} samples in predictions while {len(labels)} samples in labels!")
            labels = labels[:min(len(preds), len(labels))]
            preds = preds[:min(len(preds), len(labels))]
        
        with open(save_path, "w", encoding="utf-8") as f:
            for i, (pred, label) in enumerate(zip(preds, labels)):
                item = {
                    "prediction": pred,
                    "target": label,
                }
                if indices is not None:
                    item["index"] = indices[i]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def rouge(preds, labels, **kwargs):
        rouge = Rouge()

        if len(preds) != len(labels):
            logger.warning(f"There are {len(preds)} samples in predictions while {len(labels)} samples in labels!")
            labels = labels[:min(len(preds), len(labels))]
            preds = preds[:min(len(preds), len(labels))]

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
    
    # def acc(eval_data=None, **kwds):
    #     if eval_data is not None:
    #         data_labels = Metric._prepare_label(eval_data)

    #     def compute_metric(indices, preds, labels=None, **kwargs):
    #         if labels is None:
    #             labels = data_labels

    #         if len(preds) != len(labels):
    #             logger.warning(f"There are {len(preds)} queries in predictions while {len(labels)} queries in labels!")

    #         labels = [labels[query_id] for query_id in indices]

    #         preds = normalize_text(preds)
    #         labels = normalize_text(labels)

    #         overlap = 0
    #         for pred, label in zip(preds, labels):
    #             if pred == label:
    #                 overlap += 1

    #         metric = {
    #             "acc": overlap / len(preds),
    #         }
    #         return metric
    #     return compute_metric
