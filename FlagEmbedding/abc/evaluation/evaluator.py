"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/evaluator.py
"""
import json
import logging
import os
import json
import pandas as pd
from typing import Dict, Optional, List, Union

from .data_loader import AbsDataLoader
from .searcher import AbsEmbedder, AbsReranker
from .utils import evaluate_metrics, evaluate_mrr

logger = logging.getLogger(__name__)


class AbsEvaluator:
    def __init__(
        self,
        data_loader: AbsDataLoader,
        overwrite: bool = False,
    ):
        self.data_loader = data_loader
        self.overwrite = overwrite
        self.dataset_dir = data_loader.dataset_dir

    def check_data_info(
        self,
        data_info: Dict[str, str],
        model_name: str,
        reranker_name: str,
        split: str,
    ):
        if data_info["dataset_dir"] != self.dataset_dir:
            raise ValueError(
                f'dataset_dir mismatch: {data_info["dataset_dir"]} vs {self.dataset_dir}'
            )
        if (
            data_info["model_name"] != model_name
            or data_info["reranker_name"] != reranker_name
        ):
            raise ValueError(
                f'model_name or reranker_name mismatch: {data_info["model_name"]} vs {model_name} or {data_info["reranker_name"]} vs {reranker_name}'
            )
        if (data_info["split"] != split):
            raise ValueError(
                f'split mismatch: {data_info["split"]} vs {split}'
            )

    def __call__(
        self,
        splits: Union[str, List[str]],
        search_results_save_dir: str,
        retriever: AbsEmbedder,
        reranker: Optional[AbsReranker] = None,
        corpus_embd_save_dir: Optional[str] = None,
        retriever_batch_size: int = 256,
        reranker_batch_size: int = 256,
        retriever_query_max_length: int = 512,
        retriever_passage_max_length: int = 512,
        reranker_max_length: int = 512,
        **kwargs,
    ):
        if isinstance(splits, str):
            splits = [splits]
        # Retrieval Stage
        no_reranker_search_results_save_dir = os.path.join(
            search_results_save_dir, str(retriever), "NoReranker"
        )
        os.makedirs(no_reranker_search_results_save_dir, exist_ok=True)

        flag = False
        for split in splits:
            split_no_reranker_search_results_save_path = os.path.join(
                no_reranker_search_results_save_dir, f"{split}.json"
            )
            if not os.path.exists(split_no_reranker_search_results_save_path) or self.overwrite:
                flag = True
                break

        no_reranker_search_results_dict = {}
        if flag:
            corpus = self.data_loader.load_corpus()

            queries_dict = {
                split: self.data_loader.load_queries(split=split)
                for split in splits
            }

            all_queries = {}
            for _, split_queries in queries_dict.items():
                all_queries.update(split_queries)

            all_no_reranker_search_results = retriever(
                corpus=corpus,
                queries=all_queries,
                corpus_embd_save_dir=corpus_embd_save_dir,
                batch_size=retriever_batch_size,
                query_max_length=retriever_query_max_length,
                passage_max_length=retriever_passage_max_length
                **kwargs,
            )

            for split in splits:
                split_queries = queries_dict[split]
                no_reranker_search_results_dict[split] = {
                    qid: all_no_reranker_search_results[qid] for qid in split_queries
                }
                split_no_reranker_search_results_save_path = os.path.join(
                    no_reranker_search_results_save_dir, f"{split}.json"
                )

                self.save_search_results(
                    model_name=str(retriever),
                    reranker_name="NoReranker",
                    search_results=no_reranker_search_results_dict[split],
                    output_path=split_no_reranker_search_results_save_path,
                    split=split,
                    dataset_dir=self.dataset_dir,
                )
        else:
            for split in splits:
                split_no_reranker_search_results_save_path = os.path.join(
                    no_reranker_search_results_save_dir, f"{split}.json"
                )
                data_info, search_results = self.load_search_results(split_no_reranker_search_results_save_path)
                
                self.check_data_info(
                    data_info=data_info,
                    model_name=str(retriever),
                    reranker_name="NoReranker",
                    split=split,
                )
                no_reranker_search_results_dict[split] = search_results
        
        # Reranking Stage
        if reranker is not None:
            reranker_search_results_save_dir = os.path.join(
                search_results_save_dir, str(retriever), str(reranker)
            )
            os.makedirs(reranker_search_results_save_dir, exist_ok=True)

            corpus = self.data_loader.load_corpus()

            queries_dict = {
                split: self.data_loader.load_queries(split=split)
                for split in splits
            }

            for split in splits:
                rerank_search_results_save_path = os.path.join(
                    reranker_search_results_save_dir, f"{split}.json"
                )

                if os.path.exists(rerank_search_results_save_path) and not self.overwrite:
                    return

                rerank_search_results = reranker(
                    corpus=corpus,
                    queries=queries_dict[split],
                    search_results=no_reranker_search_results_dict[split],
                    batch_size=reranker_batch_size,
                    max_length=reranker_max_length,
                    **kwargs,
                )

                self.save_search_results(
                    model_name=str(retriever),
                    reranker_name=str(reranker),
                    search_results=rerank_search_results,
                    output_path=rerank_search_results_save_path,
                    split=split,
                    dataset_dir=self.dataset_dir,
                )

    @staticmethod
    def save_search_results(
        model_name: str,
        reranker_name: str,
        search_results: Dict[str, Dict[str, float]],
        output_path: str,
        split: str,
        dataset_dir: str,
    ):
        data = {
            "model_name": model_name,
            "reranker_name": reranker_name,
            "dataset_dir": dataset_dir,
            "split": split,
            "search_results": search_results,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_search_results(input_path: str):
        with open(input_path, "r", encoding="utf-8") as f:
            data_info = json.load(f)

        search_results = data_info.pop("search_results")
        return data_info, search_results

    @staticmethod
    def compute_metrics(
        qrels: Dict[str, Dict[str, int]],
        search_results: Dict[str, Dict[str, float]],
        k_values: List[int],
    ):
        ndcg, _map, recall, precision = evaluate_metrics(
            qrels=qrels,
            results=search_results,
            k_values=k_values,
        )
        mrr = evaluate_mrr(
            qrels=qrels,
            results=search_results,
            k_values=k_values,
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        return scores

    def evaluate_results(
        self,
        search_results_save_dir: str,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000]
    ):
        eval_results_dict = {}

        for file in os.listdir(search_results_save_dir):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(search_results_save_dir, file)
            data_info, search_results = self.load_search_results(file_path)

            _dataset_dir = data_info['dataset_dir']
            assert _dataset_dir == self.dataset_dir, f'Mismatch dataset_dir: {_dataset_dir} vs {self.dataset_dir} in {file_path}'

            split = data_info['split']
            qrels = self.data_loader.load_qrels(split=split)

            eval_results = self.compute_metrics(
                qrels=qrels,
                search_results=search_results,
                k_values=k_values
            )

            eval_results_dict[split] = eval_results

        return eval_results_dict

    @staticmethod
    def output_eval_results_to_json(eval_results_dict: dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results_dict, f, indent=4)
        print(f"Results saved to {output_path}")

    @staticmethod
    def get_results_df(metric: str, eval_results_dict: dict):
        results_dict = {}

        for model_name, model_results in eval_results_dict.items():
            results_dict[model_name] = {}
            for reranker_name, reranker_results in model_results.items():
                results_dict[model_name][reranker_name] = {}
                for split, split_results in reranker_results.items():
                    if metric in split_results:
                        results_dict[model_name][reranker_name][split] = split_results[metric]
                    else:
                        results_dict[model_name][reranker_name][split] = None

        model_reranker_pairs = set()
        all_splits = set()
        for model_name, model_results in results_dict.items():
            for reranker_name, reranker_results in model_results.items():
                model_reranker_pairs.add((model_name, reranker_name))
                all_splits.update(reranker_results.keys())

        index = [(model, reranker) for model, reranker in model_reranker_pairs]
        multi_index = pd.MultiIndex.from_tuples(index, names=['Model', 'Reranker'])
        
        all_splits = sorted(list(all_splits))
        overall_columns = ['average'] + all_splits
        overall_df = pd.DataFrame(index=multi_index, columns=overall_columns)
        
        for model, reranker in model_reranker_pairs:
            for split in all_splits:
                if model in results_dict and reranker in results_dict[model] and split in results_dict[model][reranker]:
                    overall_df.loc[(model, reranker), split] = results_dict[model][reranker][split]
                else:
                    overall_df.loc[(model, reranker), split] = None
            if overall_df.loc[(model, reranker), all_splits].isnull().any():
                overall_df.loc[(model, reranker), 'average'] = None
            else:
                overall_df.loc[(model, reranker), 'average'] = overall_df.loc[(model, reranker), all_splits].mean()

        return overall_df

    @staticmethod
    def output_eval_results_to_markdown(eval_results_dict: dict, output_path: str, metrics: Union[List[str], str]):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if isinstance(metrics, str):
            metrics = [metrics]

        with open(output_path, 'w', encoding='utf-8') as f:
            for metric in metrics:
                f.write(f"## {metric}\n\n")
                results_df = AbsEvaluator.get_results_df(metric, eval_results_dict)
                max_index = dict(results_df.idxmax(axis=0))
                splits = results_df.columns
                f.write(f"| Model | Reranker | {' | '.join(splits)} |\n")
                f.write(f"| :---- | :---- | {' | '.join([':---:' for _ in splits])} |\n")
                for i, row in results_df.iterrows():
                    line = f"| {i[0]} | {i[1]} | "
                    for s, v in row.items():
                        if v is None:
                            line += "- | "
                        else:
                            if i != max_index[s]:
                                line += f'{v*100:.3f} | '
                            else:
                                line += f'**{v*100:.3f}** | '
                    f.write(line + "\n")
                f.write("\n")
        print(f"Results saved to {output_path}")
