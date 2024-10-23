import os
from FlagEmbedding import FlagAutoReranker


def test_base_multi_devices():
    model = FlagAutoReranker.from_finetuned(
        'BAAI/bge-reranker-v2.5-gemma2-lightweight',
        use_fp16=True,
        query_instruction_for_rerank="A: ",
        passage_instruction_for_rerank="B: ",
        trust_remote_code=True,
        devices=["cuda:3"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )
    
    pairs = [
        ["What is the capital of France?", "Paris is the capital of France."],
        ["What is the capital of France?", "The population of China is over 1.4 billion people."],
        ["What is the population of China?", "Paris is the capital of France."],
        ["What is the population of China?", "The population of China is over 1.4 billion people."]
    ] * 100
    
    scores = model.compute_score(pairs, cutoff_layers=[28], compress_ratio=2, compress_layers=[24, 40])
    
    print(scores[:4])


if __name__ == '__main__':
    test_base_multi_devices()
    
    print("--------------------------------")
    print("Expected Output:")
    print("[25.375, 8.734375, 9.8359375, 26.15625]")
