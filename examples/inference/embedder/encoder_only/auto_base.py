from FlagEmbedding import FlagAutoModel


def test_auto_base():
    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-small-en-v1.5',
        normalize_embeddings=True,
        use_fp16=True,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: "
    )
    
    queries = [
        "What is the capital of France?",
        "What is the population of China?",
    ]
    passages = [
        "Paris is the capital of France.",
        "Beijing is the capital of China.",
        "The population of China is over 1.4 billion people."
    ]
    
    queries_embeddings = model.encode_queries(queries)
    passages_embeddings = model.encode_corpus(passages)
    
    cos_scores = queries_embeddings @ passages_embeddings.T
    print(cos_scores)


if __name__ == '__main__':
    test_auto_base()
