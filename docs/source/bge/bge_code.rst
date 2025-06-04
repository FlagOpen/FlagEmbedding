BGE-Code-v1
===========

**`BGE-Code-v1 <https://huggingface.co/BAAI/bge-code-v1>`_** is an LLM-based code embedding model that supports code retrieval, text retrieval, and multilingual retrieval. It primarily demonstrates the following capabilities:
- Superior Code Retrieval Performance: The model demonstrates exceptional code retrieval capabilities, supporting natural language queries in both English and Chinese, as well as 20 programming languages.
- Robust Text Retrieval Capabilities: The model maintains strong text retrieval capabilities comparable to text embedding models of similar scale.
- Extensive Multilingual Support: BGE-Code-v1 offers comprehensive multilingual retrieval capabilities, excelling in languages such as English, Chinese, Japanese, French, and more.

+-------------------------------------------------------------------+-----------------+------------+--------------+----------------------------------------------------------------------------------------------------+
|                                  Model                            |    Language     | Parameters |  Model Size  |                                            Description                                             |
+===================================================================+=================+============+==============+====================================================================================================+
| `BAAI/bge-code-v1 <https://huggingface.co/BAAI/bge-code-v1>`_       |     Multilingual     |    1.5B    |    6.18 GB   | SOTA code retrieval model, with exceptional multilingual text retrieval performance as well |
+-------------------------------------------------------------------+-----------------+------------+--------------+----------------------------------------------------------------------------------------------------+


.. code:: python
    from FlagEmbedding import FlagLLMModel

    queries = [
        "Delete the record with ID 4 from the 'Staff' table.", 
        'Delete all records in the "Livestock" table where age is greater than 5'
    ]
    documents = [
        "DELETE FROM Staff WHERE StaffID = 4;",
        "DELETE FROM Livestock WHERE age > 5;"
    ]
    
    model = FlagLLMModel('BAAI/bge-code-v1', 
                        query_instruction_format="<instruct>{}\n<query>{}",
                        query_instruction_for_retrieval="Given a question in text, retrieve SQL queries that are appropriate responses to the question.",
                        trust_remote_code=True,
                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    embeddings_1 = model.encode_queries(queries)
    embeddings_2 = model.encode_corpus(documents)
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
