import os
from FlagEmbedding import FlagModel


def test_base():
    model = FlagModel(
        'BAAI/bge-small-en-v1.5',
        normalize_embeddings=True,
        use_fp16=True,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        query_instruction_format="{}{}",
        pooling_method='cls',
        cache_dir=os.getenv('HF_HOME', None),
    )
    
    queries = [
        "What is the capital of France?",
        "What is the population of China?",
    ]
    passages = [
        "Paris is the capital of France.",
        "The population of China is over 1.4 billion people."
    ]
    
    queries_embeddings = model.encode_queries(queries)
    passages_embeddings = model.encode_corpus(passages)
    
    cos_scores = queries_embeddings @ passages_embeddings.T
    print(cos_scores)


if __name__ == '__main__':
    test_base()
