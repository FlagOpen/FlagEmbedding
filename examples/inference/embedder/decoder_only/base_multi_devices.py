import os
from FlagEmbedding import FlagLLMModel


def test_base_multi_devices():
    model = FlagLLMModel(
        'BAAI/bge-multilingual-gemma2',
        normalize_embeddings=True,
        use_fp16=True,
        query_instruction_for_retrieval="Given a question, retrieve passages that answer the question.",
        query_instruction_format="<instruct>{}\n<query>{}",
        devices=["cuda:0", "cuda:1"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
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
