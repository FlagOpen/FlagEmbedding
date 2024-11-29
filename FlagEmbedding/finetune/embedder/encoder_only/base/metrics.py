import numpy as np
import pandas as pd

def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.
    
    This function computes the average prescision at k between two lists of
    items.
    
    Parameters
    ----------
    actual : list of float
    predicted : numpy.ndarray
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.
    
    This function computes the mean average prescision at k between two lists
    of lists of items.
    
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def mean_average_precision_at_k(ground_truth, sorted_indices, k):
    """
    Calculate MAP@k for cases with a single ground truth per query.
    Skip queries with NaN ground truth.

    Parameters:
    - ground_truth: A list where each entry is a single relevant item ID or NaN.
    - sorted_indices: A list of lists, where each inner list contains sorted retrieved item IDs for a query.
    - k: The cutoff value for the top-k retrieved items.

    Returns:
    - MAP@k across valid queries.
    """
    valid_aps = []

    for gt, retrieved in zip(ground_truth, sorted_indices):
        # if not np.isnan(gt):
        if pd.notna(gt):
            # Get first k predictions
            retrieved_at_k = retrieved[:k]
            # Find where the ground truth appears in the top k predictions
            gt_positions = np.where(retrieved_at_k == gt)[0]
            if len(gt_positions) > 0:
                # Add 1 because position is 0-based
                rank = gt_positions[0] + 1
                valid_aps.append(1 / rank)
            else:
                valid_aps.append(0)

    return np.mean(valid_aps) if valid_aps else np.nan

