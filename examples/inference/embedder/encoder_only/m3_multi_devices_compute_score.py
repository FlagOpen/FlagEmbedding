import os
from FlagEmbedding import BGEM3FlagModel


def test_m3_multi_devices():
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
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

    print("--------------------------------")
    print("Expected Output:")
    print("{'colbert': [0.7798609733581543, 0.7897368669509888], 'sparse': [0.1956787109375, 0.1802978515625], 'dense': [0.6259765625, 0.67822265625], 'sparse+dense': [0.5266770720481873, 0.5633169412612915], 'colbert+sparse+dense': [0.6367570757865906, 0.6617604494094849]}")
    print("{'colbert': [0.4524071514606476, 0.4619773030281067], 'sparse': [0.0, 0.0087890625], 'dense': [0.349853515625, 0.34765625], 'sparse+dense': [0.2691181004047394, 0.269456148147583], 'colbert+sparse+dense': [0.34880897402763367, 0.3531610071659088]}")
