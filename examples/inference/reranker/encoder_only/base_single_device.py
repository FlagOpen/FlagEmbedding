import os
from FlagEmbedding import FlagReranker


def test_base_multi_devices():
    model = FlagReranker(
        'BAAI/bge-reranker-large',
        use_fp16=True,
        devices=["cuda:3"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
        cache_dir='/share/shared_models'
        # cache_dir=os.getenv('HF_HUB_CACHE', None),
    )
    
    pairs = [
        ["What is the capital of France?", "Paris is the capital of France."],
        ["What is the capital of France?", "The population of China is over 1.4 billion people."],
        ["What is the population of China?", "Paris is the capital of France."],
        ["What is the population of China?", "The population of China is over 1.4 billion people."]
    ] * 100
    
    scores = model.compute_score(pairs)
    
    print(scores[:4])


if __name__ == '__main__':
    test_base_multi_devices()
    
    print("--------------------------------")
    print("Expected Output:")
    print("[7.9765625, -6.859375, -7.15625, 5.44921875]")
