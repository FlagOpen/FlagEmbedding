BGE-EN-ICL
==========

BGE-EN-ICL is the new SoTA embedding model in BGE series with capabilities:
- In-context learning ability: By providing few-shot examples in the query, it can significantly enhance the model's ability to handle new tasks.
- Outstanding performance: The model has achieved state-of-the-art (SOTA) performance on MTEB and AIR-Bench.

+-------------------------------------------------------------------+-----------------+------------+--------------+----------------------------------------------------------------------------------------------------+
|                                  Model                            |    Language     | Parameters |  Model Size  |                                            Description                                             |
+===================================================================+=================+============+==============+====================================================================================================+
| `BAAI/bge-en-icl <https://huggingface.co/BAAI/bge-en-icl>`_       |     English     |    7.1B    |    28.5 GB   | In-context learning capabilities, fully leverage the model's potential based on a few shot examples|
+-------------------------------------------------------------------+-----------------+------------+--------------+----------------------------------------------------------------------------------------------------+



Usage
-----

.. code:: python

    from FlagEmbedding import FlagICLModel

    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
    ]

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

    queries = ["how much protein should a female eat", "summit define"]

    model = FlagICLModel('BAAI/bge-en-icl', 
                         examples_for_task=examples,  # set `examples_for_task=None` to use model without examples
                         examples_instruction_format="<instruct>{}\n<query>{}\n<response>{}") # specify the format to use examples_for_task

    embeddings_1 = model.encode_queries(queries)
    embeddings_2 = model.encode_corpus(documents)
    similarity = embeddings_1 @ embeddings_2.T

    print(similarity)