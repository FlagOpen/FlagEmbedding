import json
import logging
import os
import json
import pandas as pd
from typing import Dict, Optional, List, Union

from FlagEmbedding.abc.evaluation.evaluator import AbsEvaluator
from FlagEmbedding.abc.evaluation.data_loader import AbsDataLoader
from FlagEmbedding.abc.evaluation.searcher import AbsEmbedder, AbsReranker

class BEIREvaluator(AbsEvaluator):
    def __init__(
        self,
        data_loader: AbsDataLoader,
        overwrite: bool = False,
    ):
        self.data_loader = data_loader
        self.overwrite = overwrite
        self.dataset_dir = data_loader.dataset_dir

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
        dataset_name = self.data_loader.dataset_name
        sub_dataset_names = self.data_loader.sub_dataset_names
        split = self.data_loader.split
        if isinstance(splits, str):
            splits = [splits]
        # Retrieval Stage
        no_reranker_search_results_save_dir = os.path.join(
            search_results_save_dir, str(retriever), "NoReranker", dataset_name
        )
        os.makedirs(no_reranker_search_results_save_dir, exist_ok=True)
        if corpus_embd_save_dir is not None: corpus_embd_save_dir = os.path.join(corpus_embd_save_dir, dataset_name)

        flag = False
        if sub_dataset_names is None:
            split_no_reranker_search_results_save_path = os.path.join(
                no_reranker_search_results_save_dir, f"{split}.json"
            )
            if not os.path.exists(split_no_reranker_search_results_save_path) or self.overwrite:
                flag = True
        else:
            for sub_dataset_name in sub_dataset_names:
                split_no_reranker_search_results_save_path = os.path.join(
                    no_reranker_search_results_save_dir, f"{sub_dataset_name}-{split}.json"
                )
                if not os.path.exists(split_no_reranker_search_results_save_path) or self.overwrite:
                    flag = True
                    break

        no_reranker_search_results_dict = {}
        if flag:
            if sub_dataset_names is None:
                corpus = self.data_loader.load_corpus(sub_dataset_name=sub_dataset_names)

                queries_dict = {
                    split: self.data_loader.load_queries(sub_dataset_name=sub_dataset_names)
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
                    passage_max_length=retriever_passage_max_length,
                    **kwargs,
                )

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
                for sub_dataset_name in sub_dataset_names:
                    corpus = self.data_loader.load_corpus(sub_dataset_name=sub_dataset_name)

                    queries_dict = {
                        split: self.data_loader.load_queries(sub_dataset_name=sub_dataset_name)
                    }

                    all_queries = {}
                    for _, split_queries in queries_dict.items():
                        all_queries.update(split_queries)

                    all_no_reranker_search_results = retriever(
                        corpus=corpus,
                        queries=all_queries,
                        corpus_embd_save_dir=None if corpus_embd_save_dir is None else os.path.join(corpus_embd_save_dir, sub_dataset_name),
                        batch_size=retriever_batch_size,
                        query_max_length=retriever_query_max_length,
                        passage_max_length=retriever_passage_max_length,
                        **kwargs,
                    )

                    split_queries = queries_dict[split]
                    no_reranker_search_results_dict[f"{sub_dataset_name}-{split}"] = {
                        qid: all_no_reranker_search_results[qid] for qid in split_queries
                    }
                    split_no_reranker_search_results_save_path = os.path.join(
                        no_reranker_search_results_save_dir, f"{sub_dataset_name}-{split}.json"
                    )

                    self.save_search_results(
                        model_name=str(retriever),
                        reranker_name="NoReranker",
                        search_results=no_reranker_search_results_dict[f"{sub_dataset_name}-{split}"],
                        output_path=split_no_reranker_search_results_save_path,
                        split=f"{sub_dataset_name}-{split}",
                        dataset_dir=self.dataset_dir,
                    )

        else:
            if sub_dataset_names is None:
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
            else:
                for sub_dataset_name in sub_dataset_names:
                    split_no_reranker_search_results_save_path = os.path.join(
                        no_reranker_search_results_save_dir, f"{sub_dataset_name}-{split}.json"
                    )
                    data_info, search_results = self.load_search_results(split_no_reranker_search_results_save_path)
                    
                    self.check_data_info(
                        data_info=data_info,
                        model_name=str(retriever),
                        reranker_name="NoReranker",
                        split=f"{sub_dataset_name}-{split}",
                    )
                    no_reranker_search_results_dict[f"{sub_dataset_name}-{split}"] = search_results
        retriever_eval_results = self.evaluate_results(no_reranker_search_results_save_dir)
        self.output_eval_results_to_json(retriever_eval_results, os.path.join(no_reranker_search_results_save_dir, 'eval.json'))
        
        # Reranking Stage
        if reranker is not None:
            reranker_search_results_save_dir = os.path.join(
                search_results_save_dir, str(retriever), str(reranker), dataset_name
            )
            os.makedirs(reranker_search_results_save_dir, exist_ok=True)

            if sub_dataset_names is None:
                corpus = self.data_loader.load_corpus(sub_dataset_name=sub_dataset_names)

                queries_dict = {
                    split: self.data_loader.load_queries(sub_dataset_name=sub_dataset_names)
                }

                rerank_search_results_save_path = os.path.join(
                    reranker_search_results_save_dir, f"{split}.json"
                )

                if os.path.exists(rerank_search_results_save_path) and not self.overwrite:
                    pass
                else:
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
            else:
                for sub_dataset_name in sub_dataset_names:
                    corpus = self.data_loader.load_corpus(sub_dataset_name=sub_dataset_name)

                    queries_dict = {
                        split: self.data_loader.load_queries(sub_dataset_name=sub_dataset_name)
                    }

                    rerank_search_results_save_path = os.path.join(
                        reranker_search_results_save_dir, f"{sub_dataset_name}-{split}.json"
                    )

                    if os.path.exists(rerank_search_results_save_path) and not self.overwrite:
                        continue

                    rerank_search_results = reranker(
                        corpus=corpus,
                        queries=queries_dict[split],
                        search_results=no_reranker_search_results_dict[f"{sub_dataset_name}-{split}"],
                        batch_size=reranker_batch_size,
                        max_length=reranker_max_length,
                        **kwargs,
                    )

                    self.save_search_results(
                        model_name=str(retriever),
                        reranker_name=str(reranker),
                        search_results=rerank_search_results,
                        output_path=rerank_search_results_save_path,
                        split=f"{sub_dataset_name}-{split}",
                        dataset_dir=self.dataset_dir,
                    )

            reranker_eval_results = self.evaluate_results(reranker_search_results_save_dir)
            self.output_eval_results_to_json(reranker_eval_results, os.path.join(reranker_search_results_save_dir, 'eval.json'))