generate_train_answer = """Please generate a brief answer to the given query according to the reference passage.

Query: {query}

Reference passage: {passage}

Answer: """

generate_train_query = """Please generate a concise query from the following corpus.

Corpus: {passage}

Query: """

generate_train_query_type2 = """Generate a concise query using the key terms based on the following corpus.

Corpus: {passage}

Concise query: """

#  The query is a user query and should be short.