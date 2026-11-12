generate_prompt = """Generate a brief answer to this query.

Query: {query}
Answer: """

generate_passage_prompt = """Generate a passage that may contain an answer to this query.

Query: {query}
Passage: """

# llama_query_generate_prompt = """Formulate a question that can be answered using information from the passage. The question should be detailed.
# Passage: {passage}
# question:"""

llama_query_generate_prompt = """Formulate a question from the following passage, don't provide any information about this passage in the question.
Passage: {passage}
question:"""

# llama_answer_generate_prompt = """Please generate an answer for the following query.
# Query: {query}
# Answer:"""

llama_answer_generate_prompt = """Please generate a brief answer for the following query using 200 words, don't try to use any examples.
Query: {query}
Brief Answer:"""

llama_generate_train_answer_prompt = """Please generate a brief answer to the given query according to the reference passage.

Query: {query}

Reference passage: {passage}

Answer: """

###########

llama_query_generate_prompt_arguana = """Formulate a refutation to the following claim, don't provide any information about this passage in the refutation. Please generate directly without and explanation.
Claim: {passage}
Refutation:"""

llama_query_generate_prompt_treccovid = """Formulate a question from the following COVID-19 passage, don't provide any information about this passage in the question. Please generate directly without and explanation.
Passage: {passage}
Question:"""

llama_query_generate_prompt_nfcorpus = """Formulate a question from the following passage, don't provide any information about this passage in the question. Please generate directly without and explanation.
Passage: {passage}
Question:"""

llama_query_generate_prompt_nq = """Formulate a question from the Wikipedia following passage, don't provide any information about this passage in the question. Please generate directly without and explanation.
Passage: {passage}
Question:"""

llama_query_generate_prompt_hotpotqa = """Formulate a multi-hop question from the following passage, don't provide any information about this passage in the question. Please generate directly without and explanation.
Passage: {passage}
Multi-hop question:"""

llama_query_generate_prompt_fiqa = """Formulate a financial question from the following user reply, don't provide any information about this user reply in the question. Please generate directly without and explanation.
User reply: {passage}
Financial question:"""

llama_query_generate_prompt_touche = """Formulate a question from the following detailed and persuasive argument, don't provide any information about this argument in the question. Please generate directly without and explanation.
Argument: {passage}
Question:"""

llama_query_generate_prompt_cqa = """Formulate a duplicate question from the following detailed question description from Stackexchange, don't provide any information about this question description in the duplicate question. Please generate directly without and explanation.
Question description: {passage}
Duplicate question:"""

llama_query_generate_prompt_quora = """Formulate a duplicate question from the following question, don't provide any information about this question in the duplicate question. Please generate directly without and explanation.
Question: {passage}
Duplicate question:"""

llama_query_generate_prompt_dbpedia = """Formulate a question from the following entity description from DBPedia, don't provide any information about this description in the question. Please generate directly without and explanation.
Description: {passage}
question:"""

llama_query_generate_prompt_scidocs = """Formulate a scientific paper title from the following paper abstract that are cited by the given paper, don't provide any information about this passage in the scientific paper title. Please generate directly without and explanation.
Paper abstract: {passage}
Scientific paper title:"""

llama_query_generate_prompt_fever = """Formulate a claim from the following document, don't provide any information about this document in the claim. Please generate directly without and explanation.
Document: {passage}
Claim:"""

llama_query_generate_prompt_climatefever = """Formulate a claim about climate change from the following document, don't provide any information about this document in the claim. Please generate directly without and explanation.
Document: {passage}
Claim about climate change:"""

llama_query_generate_prompt_scifact = """Formulate a scientific claim from the following document, don't provide any information about this document in the claim. Please generate directly without and explanation.
Document: {passage}
Scientific claim:"""