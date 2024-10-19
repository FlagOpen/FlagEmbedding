import os
from FlagEmbedding import BGEM3FlagModel


def test_m3_single_device():
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        devices="cuda:0",   # if you don't have a GPU, you can use "cpu"
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
    test_m3_single_device()

    print("--------------------------------")
    print("Expected Output:")
    print("{'colbert': [0.7798250317573547, 0.7899274826049805], 'sparse': [0.195556640625, 0.180419921875], 'dense': [0.6259765625, 0.67822265625], 'sparse+dense': [0.5266488790512085, 0.5633450746536255], 'colbert+sparse+dense': [0.6367254853248596, 0.6618592143058777]}")
    print("{'colbert': [0.4524373412132263, 0.46213820576667786], 'sparse': [0.0, 0.0088043212890625], 'dense': [0.349609375, 0.34765625], 'sparse+dense': [0.2689302861690521, 0.26945966482162476], 'colbert+sparse+dense': [0.34871599078178406, 0.3532329499721527]}")
