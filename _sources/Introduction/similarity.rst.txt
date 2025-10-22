Similarity
==========

A primary goal of retrieval is to find the most relevant documents in response to a user's query. 
One of the core components of this process is measuring similarity between the query and candidates.
Similarity metrics quantify how closely related two pieces of data are, and guide the retrieval system in ranking results.

Jaccard Similarity
------------------

.. math::

    J(A,B)=\frac{|A\cap B|}{|A\cup B|}

The Jaccard similarity or Jaccard index is commonly used for set-based similarity, particularly in binary data (e.g., whether a term appears in a document or not). 
It is calculated as the size of the intersection of two sets divided by the size of their union. 
In information retrieval, it's often used to compare sets of keywords or phrases, with higher values indicating more similarity.

Euclidean Distance
------------------

.. math::

    d(A, B) = \|A-B\|_2 = \sqrt{\sum_{i=1}^n (A_i-B_i)^2}

Euclidean distance measures the straight-line distance between two points in a vector space. 
In IR, this can be used to assess the difference between document or query vectors. 
A smaller distance indicates greater similarity. 
This metric is intuitive but can sometimes be sensitive to the scale of the data, especially in high-dimensional spaces like text embeddings.

Cosine Similarity
-----------------

.. math::

    \cos(\theta)=\frac{A\cdot B}{\|A\|\|B\|}

Cosine similarity is one of the most widely used metrics in information retrieval, especially for text. 
It measures the cosine of the angle between two vectors in a multi-dimensional space (typically representing term frequency vectors of documents and queries). 
If the cosine similarity is closer to 1, the vectors are more similar. 
A value of 0 indicates orthogonality, meaning no similarity. 
It's a simple yet effective measure for text-based retrieval, as it considers the orientation but not the magnitude of vectors.

Dot Product
-----------

Coordinate definition:
.. math::

    A\cdot B = \sum_{i=1}^{i=n}A_i B_i

Geometric definition:
.. math::

    A\cdot B = \|A\|\|B\|\cos(\theta)

The dot product between two vectors provides a measure of how similar the vectors are in terms of direction and magnitude. 
In information retrieval, the dot product is often used in vector space models, particularly when dealing with pre-trained word or sentence embeddings. 
A higher dot product indicates that the query and document are closely aligned in the vector space.

