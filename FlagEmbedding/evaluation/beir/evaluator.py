import json
import logging
import os
import json
from typing import Dict, Optional, List, Union

from FlagEmbedding.abc.evaluation import AbsEvaluator, EvalRetriever, EvalReranker

logger = logging.getLogger(__name__)


class BEIREvaluator(AbsEvaluator):
    """
    Evaluator class of BEIR 
    """
    def check_data_info(
        self,
        data_info: Dict[str, str],
        model_name: str,
        reranker_name: str,
        split: str,
        dataset_name: Optional[str] = None,
        sub_dataset_name: Optional[str] = None,
    ):
        """Check the validity of data info.

        Args:
            data_info (Dict[str, str]): The loaded data info to be check.
            model_name (str): Name of model used.
            reranker_name (str): Name of reranker used.
            split (str): Split used in searching.
            dataset_name (Optional[str], optional): Name of dataset used. Defaults to None.
            sub_dataset_name (Optional[str], optional): Name of the sub-dataset. Defaults to ``None``.

        Raises:
            ValueError: eval_name mismatch
            ValueError: model_name or reranker_name mismatch
            ValueError: split mismatch
            ValueError: dataset_name mismatch
            ValueError: sub_dataset_name mismatch
        """
        if data_info["eval_name"] != self.eval_name:
            raise ValueError(
                f'eval_name mismatch: {data_info["eval_name"]} vs {self.eval_name}'
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
        if dataset_name is not None and data_info["dataset_name"] != dataset_name:
            raise ValueError(
                f'dataset_name mismatch: {data_info["dataset_name"]} vs {dataset_name}'
            )
        if sub_dataset_name is not None and data_info["sub_dataset_name"] != sub_dataset_name:
            raise ValueError(
                f'sub_dataset_name mismatch: {data_info["sub_dataset_name"]} vs {sub_dataset_name}'
            )

    def __call__(
        self,
        splits: Union[str, List[str]],
        search_results_save_dir: str,
        retriever: EvalRetriever,
        reranker: Optional[EvalReranker] = None,
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000],
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        sub_dataset_name = None
        sub_dataset_names = self.data_loader.available_sub_dataset_names(dataset_name=dataset_name)
        # Check Splits
        checked_splits = self.data_loader.check_splits(splits, dataset_name=dataset_name)
        if len(checked_splits) == 0:
            logger.warning(f"{splits} not found in the dataset. Skipping evaluation.")
            return
        splits = checked_splits

        if sub_dataset_names is None:
            if dataset_name is not None:
                save_name = f"{dataset_name}-" + "{split}.json"
                corpus_embd_save_dir = os.path.join(corpus_embd_save_dir, str(retriever), dataset_name)
            else:
                save_name = "{split}.json"

            # Retrieval Stage
            no_reranker_search_results_save_dir = os.path.join(
                search_results_save_dir, str(retriever), "NoReranker"
            )
            os.makedirs(no_reranker_search_results_save_dir, exist_ok=True)

            flag = False
            for split in splits:
                split_no_reranker_search_results_save_path = os.path.join(
                    no_reranker_search_results_save_dir, save_name.format(split=split)
                )
                if not os.path.exists(split_no_reranker_search_results_save_path) or self.overwrite:
                    flag = True
                    break

            no_reranker_search_results_dict = {}
            if flag:
                corpus = self.data_loader.load_corpus(dataset_name=dataset_name)

                queries_dict = {
                    split: self.data_loader.load_queries(dataset_name=dataset_name, split=split)
                    for split in splits
                }

                all_queries = {}
                for _, split_queries in queries_dict.items():
                    all_queries.update(split_queries)

                all_no_reranker_search_results = retriever(
                    corpus=corpus,
                    queries=all_queries,
                    corpus_embd_save_dir=corpus_embd_save_dir,
                    ignore_identical_ids=ignore_identical_ids,
                    **kwargs,
                )

                for split in splits:
                    split_queries = queries_dict[split]
                    no_reranker_search_results_dict[split] = {
                        qid: all_no_reranker_search_results[qid] for qid in split_queries
                    }
                    split_no_reranker_search_results_save_path = os.path.join(
                        no_reranker_search_results_save_dir, save_name.format(split=split)
                    )
                    self.save_search_results(
                        eval_name=self.eval_name,
                        model_name=str(retriever),
                        reranker_name="NoReranker",
                        search_results=no_reranker_search_results_dict[split],
                        output_path=split_no_reranker_search_results_save_path,
                        split=split,
                        dataset_name=dataset_name,
                        sub_dataset_name=sub_dataset_name,
                    )
            else:
                for split in splits:
                    split_no_reranker_search_results_save_path = os.path.join(
                        no_reranker_search_results_save_dir, save_name.format(split=split)
                    )
                    data_info, search_results = self.load_search_results(split_no_reranker_search_results_save_path)
                    
                    self.check_data_info(
                        data_info=data_info,
                        model_name=str(retriever),
                        reranker_name="NoReranker",
                        split=split,
                        dataset_name=dataset_name,
                        sub_dataset_name=sub_dataset_name,
                    )
                    no_reranker_search_results_dict[split] = search_results
            retriever.stop_multi_process_pool()
            eval_results_save_path = os.path.join(no_reranker_search_results_save_dir, 'EVAL', 'eval_results.json')
            retriever_eval_results = self.evaluate_results(no_reranker_search_results_save_dir, k_values=k_values)
            self.output_eval_results_to_json(retriever_eval_results, eval_results_save_path)

            # Reranking Stage
            if reranker is not None:
                reranker_search_results_save_dir = os.path.join(
                    search_results_save_dir, str(retriever), str(reranker)
                )
                os.makedirs(reranker_search_results_save_dir, exist_ok=True)

                corpus = self.data_loader.load_corpus(dataset_name=dataset_name)

                queries_dict = {
                    split: self.data_loader.load_queries(dataset_name=dataset_name, split=split)
                    for split in splits
                }

                for split in splits:
                    rerank_search_results_save_path = os.path.join(
                        reranker_search_results_save_dir, save_name.format(split=split)
                    )

                    if os.path.exists(rerank_search_results_save_path) and not self.overwrite:
                        continue

                    rerank_search_results = reranker(
                        corpus=corpus,
                        queries=queries_dict[split],
                        search_results=no_reranker_search_results_dict[split],
                        ignore_identical_ids=ignore_identical_ids,
                        **kwargs,
                    )

                    self.save_search_results(
                        eval_name=self.eval_name,
                        model_name=str(retriever),
                        reranker_name=str(reranker),
                        search_results=rerank_search_results,
                        output_path=rerank_search_results_save_path,
                        split=split,
                        dataset_name=dataset_name,
                        sub_dataset_name=sub_dataset_name,
                    )
                eval_results_save_path = os.path.join(reranker_search_results_save_dir, 'EVAL', 'eval_results.json')
                reranker_eval_results = self.evaluate_results(reranker_search_results_save_dir, k_values=k_values)
                self.output_eval_results_to_json(reranker_eval_results, eval_results_save_path)
        else:
            for sub_dataset_name in sub_dataset_names:
                if dataset_name is not None:
                    save_name = f"{dataset_name}-{sub_dataset_name}-" + "{split}.json"
                    corpus_embd_save_dir = os.path.join(corpus_embd_save_dir, str(retriever), dataset_name, sub_dataset_name)
                else:
                    save_name = f"{sub_dataset_name}-" + "{split}.json"

                # Retrieval Stage
                no_reranker_search_results_save_dir = os.path.join(
                    search_results_save_dir, str(retriever), "NoReranker"
                )
                os.makedirs(no_reranker_search_results_save_dir, exist_ok=True)

                flag = False
                for split in splits:
                    split_no_reranker_search_results_save_path = os.path.join(
                        no_reranker_search_results_save_dir, save_name.format(split=split)
                    )
                    if not os.path.exists(split_no_reranker_search_results_save_path) or self.overwrite:
                        flag = True
                        break

                no_reranker_search_results_dict = {}
                if flag:
                    corpus = self.data_loader.load_corpus(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name)

                    queries_dict = {
                        split: self.data_loader.load_queries(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name, split=split)
                        for split in splits
                    }

                    all_queries = {}
                    for _, split_queries in queries_dict.items():
                        all_queries.update(split_queries)

                    all_no_reranker_search_results = retriever(
                        corpus=corpus,
                        queries=all_queries,
                        corpus_embd_save_dir=corpus_embd_save_dir,
                        ignore_identical_ids=ignore_identical_ids,
                        **kwargs,
                    )

                    for split in splits:
                        split_queries = queries_dict[split]
                        no_reranker_search_results_dict[split] = {
                            qid: all_no_reranker_search_results[qid] for qid in split_queries
                        }
                        split_no_reranker_search_results_save_path = os.path.join(
                            no_reranker_search_results_save_dir, save_name.format(split=split)
                        )

                        self.save_search_results(
                            eval_name=self.eval_name,
                            model_name=str(retriever),
                            reranker_name="NoReranker",
                            search_results=no_reranker_search_results_dict[split],
                            output_path=split_no_reranker_search_results_save_path,
                            split=split,
                            dataset_name=dataset_name,
                            sub_dataset_name=sub_dataset_name,
                        )
                else:
                    for split in splits:
                        split_no_reranker_search_results_save_path = os.path.join(
                            no_reranker_search_results_save_dir, save_name.format(split=split)
                        )
                        data_info, search_results = self.load_search_results(split_no_reranker_search_results_save_path)
                        
                        self.check_data_info(
                            data_info=data_info,
                            model_name=str(retriever),
                            reranker_name="NoReranker",
                            split=split,
                            dataset_name=dataset_name,
                            sub_dataset_name=sub_dataset_name,
                        )
                        no_reranker_search_results_dict[split] = search_results
                eval_results_save_path = os.path.join(no_reranker_search_results_save_dir, 'EVAL', 'eval_results.json')
                retriever_eval_results = self.evaluate_results(no_reranker_search_results_save_dir, k_values=k_values)
                self.output_eval_results_to_json(retriever_eval_results, eval_results_save_path)

                # Reranking Stage
                if reranker is not None:
                    reranker_search_results_save_dir = os.path.join(
                        search_results_save_dir, str(retriever), str(reranker)
                    )
                    os.makedirs(reranker_search_results_save_dir, exist_ok=True)

                    corpus = self.data_loader.load_corpus(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name)

                    queries_dict = {
                        split: self.data_loader.load_queries(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name, split=split)
                        for split in splits
                    }

                    for split in splits:
                        rerank_search_results_save_path = os.path.join(
                            reranker_search_results_save_dir, save_name.format(split=split)
                        )

                        if os.path.exists(rerank_search_results_save_path) and not self.overwrite:
                            continue

                        rerank_search_results = reranker(
                            corpus=corpus,
                            queries=queries_dict[split],
                            search_results=no_reranker_search_results_dict[split],
                            ignore_identical_ids=ignore_identical_ids,
                            **kwargs,
                        )

                        self.save_search_results(
                            eval_name=self.eval_name,
                            model_name=str(retriever),
                            reranker_name=str(reranker),
                            search_results=rerank_search_results,
                            output_path=rerank_search_results_save_path,
                            split=split,
                            dataset_name=dataset_name,
                            sub_dataset_name=sub_dataset_name,
                        )
                    eval_results_save_path = os.path.join(reranker_search_results_save_dir, 'EVAL', 'eval_results.json')
                    reranker_eval_results = self.evaluate_results(reranker_search_results_save_dir, k_values=k_values)
                    self.output_eval_results_to_json(reranker_eval_results, eval_results_save_path)
            if reranker is not None:
                reranker.stop_multi_process_pool()
                
    def evaluate_results(
        self,
        search_results_save_dir: str,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000]
    ):
        """Compute metrics according to the results in the directory.

        Args:
            search_results_save_dir (str): Path to the search results.
            k_values (List[int], optional): Cutoffs. Defaults to :data:`[1, 3, 5, 10, 100, 1000]`.

        Returns:
            dict: Evaluation results.
        """
        eval_results_dict = {}
        cqadupstack_results = None
        cqadupstack_num = 0

        for file in os.listdir(search_results_save_dir):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(search_results_save_dir, file)
            data_info, search_results = self.load_search_results(file_path)

            _eval_name = data_info['eval_name']
            assert _eval_name == self.eval_name, f'Mismatch eval_name: {_eval_name} vs {self.eval_name} in {file_path}'

            split = data_info['split']
            dataset_name = data_info.get('dataset_name', None)
            sub_dataset_name = data_info.get('sub_dataset_name', None)
            qrels = self.data_loader.load_qrels(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name, split=split)

            eval_results = self.compute_metrics(
                qrels=qrels,
                search_results=search_results,
                k_values=k_values
            )

            if dataset_name is not None:
                if sub_dataset_name is None:
                    key = f"{dataset_name}-{split}"
                else:
                    key = f"{dataset_name}-{sub_dataset_name}-{split}"
            else:
                if sub_dataset_name is None:
                    key = split
                else:
                    key = f"{sub_dataset_name}-{split}"
            if sub_dataset_name is None:
                eval_results_dict[key] = eval_results
            else:
                if cqadupstack_results is None:
                    cqadupstack_results = eval_results
                    cqadupstack_num += 1
                else:
                    for k, v in eval_results.items():
                        cqadupstack_results[k] += v
                    cqadupstack_num += 1
        
        if cqadupstack_num > 0:
            for k in cqadupstack_results.keys():
                cqadupstack_results[k] /= cqadupstack_num
            eval_results_dict['cqadupstack-test'] = cqadupstack_results

        return eval_results_dict
    
    def save_search_results(
        self,
        eval_name: str,
        model_name: str,
        reranker_name: str,
        search_results: Dict[str, Dict[str, float]],
        output_path: str,
        split: str,
        dataset_name: Optional[str] = None,
        sub_dataset_name: Optional[str] = None,
    ):
        """Save the metadata and search results into a file.

        Args:
            eval_name (str): The experiment name of current evaluation.
            model_name (str): Name of model used.
            reranker_name (str): Name of reranker used.
            search_results (Dict[str, Dict[str, float]]): Dictionary of search results.
            output_path (str): Output path to write the results.
            split (str): Split used in searching.
            dataset_name (Optional[str], optional): Name of dataset used. Defaults to ``None``.
            sub_dataset_name (Optional[str], optional): Name of the sub-dataset. Defaults to ``None``.
        """
        data = {
            "eval_name": eval_name,
            "model_name": model_name,
            "reranker_name": reranker_name,
            "split": split,
            "dataset_name": dataset_name,
            "sub_dataset_name": sub_dataset_name,
            "search_results": search_results,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)