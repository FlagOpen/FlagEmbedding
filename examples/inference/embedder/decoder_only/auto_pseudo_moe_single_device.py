import os
from FlagEmbedding import FlagAutoModel


def test_auto_pseudo_moe_single_device():
    model_name_or_path = "geevec-ai/geevec-embeddings-1.0-lite"

    model = FlagAutoModel.from_finetuned(
        model_name_or_path,
        model_class="decoder-only-pseudo_moe",
        query_instruction_for_retrieval="Given a question, retrieve passages that answer the question.",
        query_instruction_format="Instruct: {}\nQuery: {}",
        domain_for_pseudo_moe="reasoning",
        use_fp16=False,
        use_bf16=True,
        trust_remote_code=True,
        devices="cuda:0",  # if you don't have a GPU, you can use "cpu"
        cache_dir=os.getenv("HF_HUB_CACHE", None),
    )

    queries = [
        "how much protein should a female eat",
        "summit define",
    ] * 10
    passages = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "Definition of summit for English Language Learners: the highest point of a mountain; the highest level; a meeting between leaders.",
    ] * 10

    queries_embeddings = model.encode_queries(queries)
    passages_embeddings = model.encode_corpus(passages)

    cos_scores = queries_embeddings @ passages_embeddings.T
    print(cos_scores[:2, :2])


if __name__ == "__main__":
    test_auto_pseudo_moe_single_device()

    print("--------------------------------")
    print("Expected Output:")
    print("[[0.844 0.466 ]\n [0.395 0.684 ]]")
