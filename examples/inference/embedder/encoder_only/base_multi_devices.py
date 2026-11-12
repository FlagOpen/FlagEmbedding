import os
from FlagEmbedding import FlagModel


def test_base_multi_devices():
    model = FlagModel(
        'BAAI/bge-small-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        query_instruction_format="{}{}",
        devices=["cuda:0", "cuda:1"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
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
    
    queries_embeddings = model.encode_queries(queries)
    passages_embeddings = model.encode_corpus(passages)
    
    cos_scores = queries_embeddings @ passages_embeddings.T
    print(cos_scores[:2, :2])


if __name__ == '__main__':
    test_base_multi_devices()
    
    print("--------------------------------")
    print("Expected Output:")
    print("[[0.7944 0.4492]\n [0.5806 0.801 ]]")
