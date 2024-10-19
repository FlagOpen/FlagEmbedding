import os
from FlagEmbedding import BGEM3FlagModel


def test_m3_single_devices():
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        normalize_embeddings=True,
        use_fp16=True,
        devices="cuda:0",   # if you don't have a GPU, you can use "cpu"
        pooling_method='cls',
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )
    
    queries = [
        "What is the capital of France?",
        "What is the population of China?",
    ] * 100
    passages = [
        "Paris is the capital of France.",
        "The population of China is over 1.4 billion people."
    ] * 100
    
    queries_embeddings = model.encode_queries(
        queries,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    passages_embeddings = model.encode_corpus(
        passages,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    
    dense_scores = queries_embeddings["dense_vecs"] @ passages_embeddings["dense_vecs"].T
    sparse_scores = model.compute_lexical_matching_score(
        queries_embeddings["lexical_weights"],
        passages_embeddings["lexical_weights"],
    )

    print("Dense score:\n", dense_scores[:2, :2])
    print("Sparse score:\n", sparse_scores[:2, :2])


if __name__ == '__main__':
    test_m3_single_devices()
