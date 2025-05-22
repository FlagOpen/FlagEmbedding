import random

QUERY_TYPE_DICT = {
    'dbpedia-entity': 'query',
    'arguana': 'claim',
    'climate-fever': 'claim',
    'cqadupstack': 'question',
    'fever': 'claim',
    'fiqa': 'financial question',
    'hotpotqa': 'multi-hop question',
    'msmarco': 'web search query',
    'nfcorpus': 'question',
    'nq': 'question',
    'quora': 'question',
    'scidocs': 'scientific paper title',
    'scifact': 'scientific claim',
    'webis-touche2020': 'question',
    'trec-covid': 'query',
}


PASSAGE_TYPE_DICT = {
    'dbpedia-entity': 'entity description',
    'arguana': 'document',
    'climate-fever': 'document',
    'cqadupstack': 'question description',
    'fever': 'document',
    'fiqa': 'user reply',
    'hotpotqa': 'document',
    'msmarco': 'passage',
    'nfcorpus': 'document',
    'nq': 'Wikipedia passage',
    'quora': 'question',
    'scidocs': 'paper abstract',
    'scifact': 'document',
    'webis-touche2020': 'argument',
    'trec-covid': 'document',
}

ANSWER_TYPE_DICT = {
    'dbpedia-entity': 'entity description',
    'arguana': 'document',
    'climate-fever': 'answer',
    'cqadupstack': 'question description',
    'fever': 'answer',
    'fiqa': 'answer',
    'hotpotqa': 'answer',
    'msmarco': 'answer',
    'nfcorpus': 'answer',
    'nq': 'answer',
    'quora': 'question',
    'scidocs': 'paper abstract',
    'scifact': 'answer',
    'webis-touche2020': 'answer',
    'trec-covid': 'answer',
}

QC_MISSION_DICT = {
    'dbpedia-entity': 'Given a {passage_type} and a {query_type}, predict whether the {passage_type} is relevant to the {query_type} by producing either "Yes" or "No".',
    'arguana': 'Given a {passage_type} and a {query_type} whether the {passage_type} can refute the {query_type} by producing either "Yes" or "No".',
    'climate-fever': 'Given a {passage_type} and a {query_type} whether the {passage_type} can support or refute the {query_type} by producing either "Yes" or "No".',
    'cqadupstack': 'Given a {passage_type} and a {query_type} whether the {passage_type} is a duplicate to the given {query_type} by producing either "Yes" or "No".',
    'fever': 'Given a {passage_type} and a {query_type} whether the {passage_type} can support or refute the {query_type} by producing either "Yes" or "No".',
    'fiqa': 'Given a {passage_type} and a {query_type} whether the {passage_type} can answer the {query_type} by producing either "Yes" or "No".',
    'hotpotqa': 'Given a {passage_type} and a {query_type} whether the {passage_type} can answer the {query_type} by producing either "Yes" or "No".',
    'msmarco': 'Given a {passage_type} and a {query_type} whether the {passage_type} is relevant to the {query_type} by producing either "Yes" or "No".',
    'nfcorpus': 'Given a {passage_type} and a {query_type} whether the {passage_type} can answer the {query_type} by producing either "Yes" or "No".',
    'nq': 'Given a {passage_type} and a {query_type} whether the {passage_type} can answer the {query_type} by producing either "Yes" or "No".',
    'quora': 'Given a {passage_type} and a {query_type} whether the {passage_type} is semantically equivalent to the given {query_type} by producing either "Yes" or "No".',
    'scidocs': 'Given a {passage_type} and a {query_type} whether the {passage_type} is possibly cited by the paper with the given {query_type} by producing either "Yes" or "No".',
    'scifact': 'Given a {passage_type} and a {query_type} whether the {passage_type} can support or refute the {query_type} by producing either "Yes" or "No".',
    'webis-touche2020': 'Given a {passage_type} and a {query_type} whether the {passage_type} can answer the {query_type} by producing either "Yes" or "No".',
    'trec-covid': 'Given a {passage_type} and a {query_type} whether the {passage_type} can answer the {query_type} by producing either "Yes" or "No".',
}

QC_MISSION_QUERY_DICT = {
    'dbpedia-entity': 'Does the {passage_type} is relevant to the {query_type}?',
    'arguana': 'Does the {passage_type} can refute the {query_type}?',
    'climate-fever': 'Does the {passage_type} can support or refute the {query_type}?',
    'cqadupstack': 'Does the {passage_type} is a duplicate to the given {query_type}?',
    'fever': 'Does the {passage_type} can support or refute the {query_type}?',
    'fiqa': 'Does the {passage_type} can answer the {query_type}?',
    'hotpotqa': 'Does the {passage_type} can answer the {query_type}?',
    'msmarco': 'Does the {passage_type} is relevant to the {query_type}?',
    'nfcorpus': 'Does the {passage_type} can answer the {query_type}?',
    'nq': 'Does the {passage_type} can answer the {query_type}?',
    'quora': 'Does the {passage_type} is semantically equivalent to the given {query_type}?',
    'scidocs': 'Does the {passage_type} is possibly cited by the paper with the given {query_type}?',
    'scifact': 'Does the {passage_type} can support or refute the {query_type}?',
    'webis-touche2020': 'Does the {passage_type} can answer the {query_type}?',
    'trec-covid': 'Does the {passage_type} can answer the {query_type}?',
}


def get_yes_prompt(dataset_name: str, query: str, passage: str) -> str:
    query_type = QUERY_TYPE_DICT[dataset_name]
    passage_type = PASSAGE_TYPE_DICT[dataset_name]
    judge = QC_MISSION_DICT[dataset_name].format(
        passage_type=passage_type, query_type=query_type
    )
    juede_dup = QC_MISSION_QUERY_DICT[dataset_name].format(
        passage_type=passage_type, query_type=query_type
    )
    
    prompt_template = """\
{judge}

{passage_type}: {passage}
{query_type}: {query}

{juede_dup}

Your Output:"""

    prompt = prompt_template.format(
        query_type=query_type,
        passage_type=passage_type,
        query=query,
        passage=passage,
        judge=judge,
        juede_dup=juede_dup,
    )
    return prompt

rank_prompt = """\
This is RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.
The following are {num} passages, each indicated by number identifier []. I can rank them based on their relevance to query: {query}
{passages}
The search query is: {query}
I will rank the {num} passages above based on their relevance to the search query. The passages will be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.
The ranking results of the {num} passages (only identifiers) is:"""

TASK_DICT = {
    'dbpedia-entity': 'Given a query, retrieve relevant entity descriptions from DBPedia.',
    'arguana': 'Given a claim, find documents that refute the claim.',
    'climate-fever': 'Given a claim about climate change, retrieve documents that support or refute the claim.',
    'cqadupstack': 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.',
    'fever': 'Given a claim, retrieve documents that support or refute the claim.',
    'fiqa': 'Given a financial question, retrieve user replies that best answer the question.',
    'hotpotqa': 'Given a multi-hop question, retrieve documents that can help answer the question.',
    'msmarco': 'Given a web search query, retrieve relevant passages that answer the query.',
    'nfcorpus': 'Given a question, retrieve relevant documents that best answer the question.',
    'nq': 'Given a question, retrieve Wikipedia passages that answer the question.',
    'quora': 'Given a question, retrieve questions that are semantically equivalent to the given question.',
    'scidocs': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.',
    'scifact': 'Given a scientific claim, retrieve documents that support or refute the claim.',
    'webis-touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question.',
    'trec-covid': 'Given a query on COVID-19, retrieve documents that answer the query.',
}

def get_rank_prompt(dataset_name, num, query, passages):
#     prompt = """\
# This is RankGPT, an intelligent assistant that can rank {passage_type}s based on their relevancy to the {query_type}.
# The task is: {task}
# The following are {num} {passage_type}s, each indicated by number identifier []. I can rank them based on their relevance to {query_type}: {query}
# {passages}
# The {query_type} is: {query}
# I will rank the {num} {passage_type}s above based on their relevance to the {query_type}. The {passage_type}s will be listed in descending order using identifiers, and the most relevant {passage_type}s should be listed first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.
# The ranking results of the {num} {passage_type}s (only identifiers) is:"""
#     query_type = QUERY_TYPE_DICT[dataset_name]
#     passage_type = PASSAGE_TYPE_DICT[dataset_name]
#     return prompt.format(
#         num=num,
#         query=query,
#         passages=passages,
#         query_type=query_type,
#         passage_type=passage_type,
#         task=TASK_DICT[dataset_name]
#     )

    task=TASK_DICT[dataset_name]

    messages = [
                {
                    'role': 'system',
                    'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query. \nThe relevance is judged based on the description of the task, which is: {task}"
                },
                {
                    'role': 'user',
                    'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}. \nThe relevance is judged based on the description of the task, which is: {task}"
                },
                {
                    'role': 'assistant', 'content': 'Okay, please provide the passages.'
                }
    ]
    for rank, content in enumerate(passages):
        content = content.replace('Title:  Content: ', '')
        content = content.strip()
        messages.append({'role': 'user', 'content': f"[{rank + 1}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank + 1}].'})
    messages.append({'role': 'user', 'content': f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."})

    return messages