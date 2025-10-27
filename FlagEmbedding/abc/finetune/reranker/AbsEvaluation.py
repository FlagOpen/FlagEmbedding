import logging
import numpy as np
from typing import Dict, List
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def compute_reranker_metrics(eval_group_size: int, k_values: List[int] = [1, 3, 5, 10, 20]):
    """
    Creates a compute_metrics function for reranker evaluation.

    Args:
        eval_group_size (int): Number of passages per query (1 positive + N-1 negatives).
                               The first passage is always the positive one.
        k_values (List[int]): List of k values for computing nDCG@k and Recall@k metrics.
                              Default: [1, 3, 5, 10, 20]

    Returns:
        callable: A function that takes EvalPrediction and returns a dict of metrics.
    """
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute IR metrics for reranker evaluation.

        The evaluation assumes:
        - Predictions are relevance scores from the reranker
        - Each query has eval_group_size passages (1 positive at index 0, rest negatives)
        - Labels are implicit (first passage is always relevant)

        Metrics computed:
        - Accuracy: Percentage of queries where positive passage ranks #1
        - MRR (Mean Reciprocal Rank): Average of 1/rank of the positive passage
        - nDCG@k: Normalized Discounted Cumulative Gain at various k values
        - Recall@k: Percentage of queries where positive is in top-k
        - mean_score: Average relevance score across all passages
        """
        predictions = eval_pred.predictions  # Shape: (num_examples,)

        # Group predictions by eval_group_size
        num_queries = len(predictions) // eval_group_size
        if len(predictions) % eval_group_size != 0:
            logger.warning(
                f"Number of predictions ({len(predictions)}) is not divisible by "
                f"eval_group_size ({eval_group_size}). Truncating extra predictions."
            )

        # Reshape to (num_queries, eval_group_size)
        grouped_scores = predictions[:num_queries * eval_group_size].reshape(num_queries, eval_group_size)

        # Calculate metrics
        metrics = {}

        # Accuracy: positive passage (index 0) has highest score
        predicted_ranks = np.argmax(grouped_scores, axis=1)
        accuracy = np.mean(predicted_ranks == 0)
        metrics['accuracy'] = float(accuracy)

        # MRR: Mean Reciprocal Rank of positive passage
        # Get ranking of each passage (argsort returns indices in ascending order, so reverse it)
        rankings = np.argsort(-grouped_scores, axis=1)  # Sort descending
        # Find position of index 0 (positive passage) in each ranking
        positive_positions = np.where(rankings == 0)[1] + 1  # +1 for 1-indexed ranks
        mrr = np.mean(1.0 / positive_positions)
        metrics['mrr'] = float(mrr)

        # nDCG@k and Recall@k for various k values
        for k in k_values:
            if k > eval_group_size:
                continue

            # Recall@k: Is positive passage in top-k?
            recall_at_k = np.mean(positive_positions <= k)
            metrics[f'recall@{k}'] = float(recall_at_k)

            # nDCG@k
            # For reranker with binary relevance (only 1 relevant doc):
            # DCG@k = 1/log2(rank+1) if positive is in top-k, else 0
            # IDCG@k = 1/log2(2) = 1.0 (ideal case: relevant doc at position 1)
            dcg_scores = np.where(
                positive_positions <= k,
                1.0 / np.log2(positive_positions + 1),
                0.0
            )
            idcg = 1.0  # Ideal: relevant document at position 1
            ndcg_at_k = np.mean(dcg_scores / idcg)
            metrics[f'ndcg@{k}'] = float(ndcg_at_k)

        # Mean score across all passages
        mean_score = np.mean(predictions)
        metrics['mean_score'] = float(mean_score)

        # Mean positive score and mean negative score
        positive_scores = grouped_scores[:, 0]  # First passage is positive
        negative_scores = grouped_scores[:, 1:].flatten()  # Rest are negatives
        metrics['mean_positive_score'] = float(np.mean(positive_scores))
        metrics['mean_negative_score'] = float(np.mean(negative_scores))

        return metrics

    return compute_metrics
