import os
from FlagEmbedding import FlagAutoModel


def test_base_multi_devices():
    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-multilingual-gemma2',
        query_instruction_for_retrieval="Given a question, retrieve passages that answer the question.",
        devices=["cuda:0", "cuda:1"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )
    
    queries = [
        "how much protein should a female eat",
        "summit define"
    ] * 100
    passages = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
    ] * 100
    
    queries_embeddings = model.encode_queries(queries)
    passages_embeddings = model.encode_corpus(passages)
    
    cos_scores = queries_embeddings @ passages_embeddings.T
    print(cos_scores[:2, :2])


if __name__ == '__main__':
    test_base_multi_devices()

    print("--------------------------------")
    print("Expected Output:")
    print("[[0.558   0.02113 ]\n [0.01643 0.526  ]]")
