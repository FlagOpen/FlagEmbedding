import os
from FlagEmbedding import FlagReranker


def test_base_multi_devices():
    model = FlagReranker(
        'BAAI/bge-reranker-large',
        use_fp16=True,
        batch_size=128,
        query_max_length=256,
        max_length=512,
        devices=["cuda:3", "cuda:4"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
        cache_dir=os.getenv('HF_HUB_CACHE', None),
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
    print("[ 7.97265625 -6.8515625  -7.15625     5.45703125]")
