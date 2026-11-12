from typing import Dict


def get_task_def_by_task_name(task_name: str) -> str:
    task_name_to_instruct: Dict[str, str] = {
        # Text-to-Code Retrieval
        ## Code Contest Retrieval
        'apps': 'Given a code contest problem description, retrieve relevant code that can help solve the problem.',
        ## Web Query to Code Retrieval
        'cosqa': 'Given a web search query, retrieve relevant code that can help answer the query.',
        ## Text-to-SQL Retrieval
        'synthetic-text2sql': 'Given a question in text, retrieve SQL queries that are appropriate responses to the question.',
        
        # Code-to-Text Retrieval
        ## Code Summary Retrieval
        'CodeSearchNet-': 'Given a piece of code, retrieve the document string that summarizes the code.',
        
        # Code-to-Code Retrieval
        ## Code Context Retrieval
        'CodeSearchNet-ccr-': 'Given a piece of code segment, retrieve the code segment that is the latter part of the code.',
        ## Similar Code Retrieval
        'codetrans-dl': 'Given a piece of code, retrieve code that is semantically equivalent to the input code.',
        'codetrans-contest': 'Given a piece of Python code, retrieve C++ code that is semantically equivalent to the input code.',
        
        # Hybrid Code Retrieval
        ## Single-turn Code QA
        'stackoverflow-qa': 'Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
        'codefeedback-st': 'Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
        ## Multi-turn Code QA
        'codefeedback-mt': 'Given a multi-turn conversation history that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
    }
    
    special_task_names = ['CodeSearchNet-ccr-', 'CodeSearchNet-']
    for special_task_name in special_task_names:
        if special_task_name in task_name:
            return task_name_to_instruct[special_task_name]
    
    return task_name_to_instruct[task_name]