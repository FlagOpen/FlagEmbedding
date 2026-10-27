Information Retrieval
=====================

What is Information Retrieval?
------------------------------

Simply put, Information Retrieval (IR) is the science of searching and retrieving information from a large collection of data based on a user's query. 
The goal of an IR system is not just to return a list of documents but to ensure that the most relevant ones appear at the top of the results.

A very straightforward example of IR is library catalog. One wants to find the book that best matches the query, but there are thousands or millions of books on the shelf.
The library's catalog system helps you find the best matches based on your search terms. 
In modern digital world, search engines and databases work in a similar way, using sophisticated algorithms and models to retrieve, rank and return the most relevant results.
And the resource categories are expanding from text to more modalities such as images, videos, 3D objects, music, etc.

IR and Embedding Model
----------------------

Traditional IR methods, like TF-IDF and BM25, rely on statistical and heuristic techniques to rank documents based on term frequency and document relevance.
These methods are efficient and effective for keyword-based search but often struggle with understanding the deeper context or semantics of the text.

.. seealso::
    
    Take a very simple example with two sentences:

    .. code:: python

        sentence_1 = "watch a play"
        sentence_2 = "play with a watch"

    Sentence 1 means going for a show/performance, which has watch as a verb and play as a noun.

    However sentence 2 means someone is interacting with a timepiece on wrist, which has play as a verb and watch as a noun.

These two sentences could be regard as very similar to each other when using the traditional IR methods though they actually have totally different semantic meaning. 
Then how could we solve this? The best answer up until now is embedding models.

Embedding models have revolutionized IR by representing text as dense vectors in a high-dimensional space, capturing the semantic meaning of words, sentences, or even entire documents. 
This allows for more sophisticated search capabilities, such as semantic search, where results are ranked based on meaning rather than simple keyword matching.