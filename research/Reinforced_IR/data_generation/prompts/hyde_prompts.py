import random


TASK_DICT = {
    'dbpedia-entity': 'Please write a passage to answer the question.',
    'arguana': 'Please write a counter argument for the passage.',
    'climate-fever': 'Please write a scientific paper passage to support/refute the claim.',
    'cqadupstack': 'Please write a duplicate passage to the question.',
    'fever': 'Please write a scientific paper passage to support/refute the claim.',
    'fiqa': 'Please write a financial article passage to answer the question.',
    'hotpotqa': 'Please write a passage to answer the question.',
    'msmarco': 'Please write a passage to answer the question.',
    'nfcorpus': 'Please write a passage to answer the question.',
    'nq': 'Please write a passage to answer the question.',
    'quora': 'Please write a duplicate passage to the question.',
    'scidocs': 'Please write a passage to cite the title.',
    'scifact': 'Please write a scientific paper passage to support/refute the claim.',
    'webis-touche2020': 'Please write a passage to answer the question.',
    'trec-covid': 'Please write a scientific paper passage to answer the question.',
}


QUERY_TYPE_DICT = {
    'dbpedia-entity': 'Question',
    'arguana': 'Passage',
    'climate-fever': 'Claim',
    'cqadupstack': 'Question',
    'fever': 'Claim',
    'fiqa': 'Question',
    'hotpotqa': 'Question',
    'msmarco': 'Question',
    'nfcorpus': 'Question',
    'nq': 'Question',
    'quora': 'Question',
    'scidocs': 'Title',
    'scifact': 'Claim',
    'webis-touche2020': 'Question',
    'trec-covid': 'Question',
}


def get_additional_info_generation_prompt(dataset_name: str, query: str) -> str:
    task = TASK_DICT[dataset_name]
    query_type = QUERY_TYPE_DICT[dataset_name]
    
    prompt_template = """{task}

{query_type}: {query}

Passage:"""

    prompt = prompt_template.format(
        task=task,
        query_type=query_type,
        query=query,
    )
    return prompt