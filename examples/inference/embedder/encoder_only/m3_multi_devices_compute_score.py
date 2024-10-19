import os
from FlagEmbedding import BGEM3FlagModel


def test_m3_multi_devices():
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        normalize_embeddings=True,
        use_fp16=True,
        devices=["cuda:0", "cuda:1"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
        pooling_method='cls',
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )
    
    queries = [
        "What is BGE M3?",
        "Defination of BM25"
    ] * 100
    passages = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"
    ] * 100
    
    sentence_pairs = list(zip(queries, passages))
    scores_dict = model.compute_score(
        sentence_pairs,
        weights_for_different_modes=[1., 0.3, 1.]
    )
    
    queries.reverse()
    sentence_pairs = list(zip(queries, passages))
    
    scores_dict_reverse = model.compute_score(
        sentence_pairs,
        weights_for_different_modes=[1., 0.3, 1.]
    )
    
    scores_dict = {
        key: value[:2]
        for key, value in scores_dict.items()
    }
    scores_dict_reverse = {
        key: value[:2]
        for key, value in scores_dict_reverse.items()
    }
    
    print(scores_dict)
    print(scores_dict_reverse)


if __name__ == '__main__':
    test_m3_multi_devices()

    # print("--------------------------------")
    # print("Expected Output:")
    # print("Dense score:")
    # print(" [[0.626  0.3477]\n [0.3496 0.678 ]]")
    # print("Sparse score:")
    # print(" [[0.19554901 0.00880432]\n [0.         0.18036556]]")
