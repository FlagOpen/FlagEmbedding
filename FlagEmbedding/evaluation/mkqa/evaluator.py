import os
from tqdm import tqdm
from typing import Dict, List, Optional

from FlagEmbedding.abc.evaluation import AbsEvaluator

from .utils.compute_metrics import evaluate_qa_recall


class MKQAEvaluator(AbsEvaluator):
    """
    The evaluator class of MKQA.
    """
    def get_corpus_embd_save_dir(
        self,
        retriever_name: str,
        corpus_embd_save_dir: Optional[str] = None,
        dataset_name: Optional[str] = None
    ):
        """Get the directory to save the corpus embedding.

        Args:
            retriever_name (str): Name of the retriever.
            corpus_embd_save_dir (Optional[str], optional): Directory to save the corpus embedding. Defaults to ``None``.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.

        Returns:
            str: The final directory to save the corpus embedding.
        """
        if corpus_embd_save_dir is not None:
            # Save the corpus embeddings in the same directory for all dataset_name
            corpus_embd_save_dir = os.path.join(corpus_embd_save_dir, retriever_name)
        return corpus_embd_save_dir

    def evaluate_results(
        self,
        search_results_save_dir: str,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000]
    ):
        """Compute the metrics and get the eval results.

        Args:
            search_results_save_dir (str): Directory that saves the search results.
            k_values (List[int], optional): Cutoffs. Defaults to ``[1, 3, 5, 10, 100, 1000]``.

        Returns:
            dict: The evaluation results.
        """
        eval_results_dict = {}

        corpus = self.data_loader.load_corpus()
        corpus_dict = {}
        for docid, data in tqdm(corpus.items(), desc="Loading corpus for evaluation"):
            title, text = data["title"], data["text"]
            corpus_dict[docid] = f"{title} {text}".strip()

        for file in os.listdir(search_results_save_dir):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(search_results_save_dir, file)
            data_info, search_results = self.load_search_results(file_path)

            _eval_name = data_info['eval_name']
            assert _eval_name == self.eval_name, f'Mismatch eval_name: {_eval_name} vs {self.eval_name} in {file_path}'

            split = data_info['split']
            dataset_name = data_info.get('dataset_name', None)
            qrels = self.data_loader.load_qrels(dataset_name=dataset_name, split=split)

            eval_results = self.compute_metrics(
                corpus_dict=corpus_dict,
                qrels=qrels,
                search_results=search_results,
                k_values=k_values
            )

            if dataset_name is not None:
                key = f"{dataset_name}-{split}"
            else:
                key = split
            eval_results_dict[key] = eval_results

        return eval_results_dict
    
    @staticmethod
    def compute_metrics(
        corpus_dict: Dict[str, str],
        qrels: Dict[str, List[str]],
        search_results: Dict[str, Dict[str, float]],
        k_values: List[int],
    ):
        """
        Compute Recall@k for QA task. The definition of recall in QA task is different from the one in IR task. Please refer to the paper of RocketQA: https://aclanthology.org/2021.naacl-main.466.pdf.
        
        Args:
            corpus_dict (Dict[str, str]): Dictionary of the corpus with doc id and contents.
            qrels (Dict[str, List[str]]): Relevances of queries and passage.
            search_results (Dict[str, Dict[str, float]]): Search results of the model to evaluate.
        
        Returns:
            dict: The model's scores of the metrics.
        """
        contexts = []
        answers = []
        top_k = max(k_values)
        for qid, doc_score_dict in search_results.items():
            doc_score_pair = sorted(doc_score_dict.items(), key=lambda x: x[1], reverse=True)
            _ctxs = [corpus_dict[docid] for docid, _ in doc_score_pair[:top_k]]
            contexts.append(_ctxs)
            answers.append(qrels[qid])

        recall = evaluate_qa_recall(contexts, answers, k_values=k_values)
        scores = {f"qa_recall_at_{k}": v for k, v in zip(k_values, recall)}

        return scores
    