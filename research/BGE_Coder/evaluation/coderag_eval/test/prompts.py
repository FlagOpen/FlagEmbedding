from typing import Dict


def get_task_def_by_task_name(task_name: str) -> str:
    task_name_to_instruct: Dict[str, str] = {
        'humaneval': 'Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
        'mbpp': 'Given a textual explanation of code functionality, retrieve the corresponding code implementation.',
        'ds1000_all_completion': 'Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
        'odex_en': 'Given a question, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
        'odex_es': 'Given a question, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
        'odex_ja': 'Given a question, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
        'odex_ru': 'Given a question, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.',
        'repoeval': 'Given a code snippet and a new function name, retrieve the implementation of the function.',
        # 'repoeval': 'Given a piece of code segment, retrieve the code segment that is the latter part of the code.',
        'swe-bench-lite': 'Given a code snippet containing a bug and a natural language description of the bug or error, retrieve code snippets that demonstrate solutions or fixes for similar bugs or errors (the desired documents).'
    }
    
    return task_name_to_instruct[task_name]