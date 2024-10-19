import os
from FlagEmbedding import FlagAutoModel


def test_icl_multi_devices():
    examples = [
        {
            'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
            'query': 'what is a virtual interface',
            'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."
        },
        {
            'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
            'query': 'causes of back pain in female for a week',
            'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."
        }
    ]
    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-en-icl',
        query_instruction_for_retrieval="Given a question, retrieve passages that answer the question.",
        examples_for_task=examples,
        examples_instruction_format="<instruct>{}\n<query>{}\n<response>{}",
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
    test_icl_multi_devices()

    print("--------------------------------")
    print("Expected Output:")
    print("[[0.579  0.2776]\n [0.2249 0.5146]]")
