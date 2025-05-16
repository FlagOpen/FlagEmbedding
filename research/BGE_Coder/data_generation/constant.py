from enum import Enum
from dataclasses import dataclass
from typing import Dict, Union, Tuple, Optional, List


class TaskType(Enum):
    # text2code
    web_code_retrieval = "Web Query to Code Retrieval"
    code_contest_retrieval = "Code Contest Retrieval"
    text2sql_retrieval = "Text to SQL Retrieval"
    error_message_retrieval = "Error Message to Code Retrieval"
    code_explanation_retrieval = "Code Explanation to Implementation Retrieval"
    api_usage_retrieval = "API Usage Description to Code Retrieval"
    bug_desc_retrieval = "Bug Description to Code Retrieval"
    pseudocode_retrieval = "Pseudocode to Code Retrieval"
    tutorial_query_retrieval = "Programming Tutorial Query to Code Example Retrieval"
    algorithm_desc_retrieval = "Algorithm Description to Code Retrieval"

    # code2text
    code_summary_retrieval = "Code Summary Retrieval"
    code_review_retrieval = "Code Review Retrieval"
    code_intent_retrieval = "Code Intent Retrieval"
    code_optimization_retrieval = "Code Optimization Retrieval"
    tutorial_retrieval = "Tutorial Retrieval"
    code_issue_discussion_retrieval = "Code Issue Discussion Retrieval"
    api_reference_retrieval = "API Reference Retrieval"
    code_walkthrough_retrieval = "Code Walkthrough Retrieval"
    code_error_explanation_retrieval = "Code Error Explanation Retrieval"
    code_to_requirement_retrieval = "Code to Requirement Retrieval"

    # code2code
    code_context_retrieval = "Code Context Retrieval"
    similar_code_retrieval = "Similar Code Retrieval"
    code_translation_retrieval = "Code Translation Retrieval"
    code_refinement_retrieval = "Code Refinement Retrieval"
    secure_code_retrieval = "Secure Code Retrieval"
    code_version_update_retrieval = "Code Version Update Retrieval"
    code_example_retrieval = "Code Example Retrieval"
    code_dependency_retrieval = "Code Dependency Retrieval"
    code_pattern_retrieval = "Code Pattern Retrieval"
    code_history_retrieval = "Code History Retrieval"
    code_integration_retrieval = "Code Integration Retrieval"
    optimized_code_retrieval = "Optimized Code Retrieval"
    code_simplification_retrieval = "Code Simplification Retrieval"
    code_modularization_retrieval = "Code Modularization Retrieval"
    code_augmentation_retrieval = "Code Augmentation Retrieval"
    error_handling_code_retrieval = "Error Handling Retrieval"
    code_documentation_retrieval = "Code Documentation Retrieval"
    library_adaptation_retrieval = "Library Adaptation Retrieval"

    # hybrid
    code_modification_retrieval = "Code Modification Retrieval"
    # single_turn_code_qa = "Single-turn Code QA"
    # multi_turn_code_qa = "Multi-turn Code QA"
    code_bug_fix_example_retrieval = "Code Bug Fix Example Retrieval"
    code_refactoring_pattern_retrieval = "Code Refactoring Pattern Retrieval"
    code_style_guideline_example_retrieval = "Code Style Guideline Example Retrieval"
    code_migration_retrieval = "Code Migration Retrieval"
    code_optimization_hybrid_retrieval = "Code Optimization Hybrid Retrieval"
    code_comparison_retrieval = "Code Comparison Retrieval"
    code_best_practices_retrieval = "Code Best Practices Retrieval"
    security_vulnerability_fix_retrieval = "Security Vulnerability Fix Retrieval"


def get_task_def_by_task_type(task_type: Union[str, TaskType]) -> Tuple[str, TaskType, str]:
    """
    Given a task type, return the main task type, task type, and task instruction.
    
    Args:
    - task_type: Union[str, TaskType]: the task type, either as a string or as a TaskType enum. Example: "web_code_retrieval" or TaskType.web_code_retrieval
    
    Returns:
    - main_task_type: str: the main task type. Example: "text2code"
    - task_type: TaskType: the task type. Example: TaskType.web_code_retrieval
    - task_instruction: str: the task instruction. Example: "Given a web search query, retrieve relevant code that can help answer the query."
    """
    
    task_type_to_instruct: Dict[TaskType, str] = {
        # text2code
        TaskType.web_code_retrieval: "Given a web search query, retrieve relevant code that can help answer the query.",
        TaskType.code_contest_retrieval: "Given a code contest problem description, retrieve relevant code that can help solve the problem.",
        TaskType.text2sql_retrieval: "Given a question in text, retrieve SQL queries that are appropriate responses to the question.",
        TaskType.error_message_retrieval: "Given an error message encountered during coding, retrieve relevant code that can help resolve the error.",
        TaskType.code_explanation_retrieval: "Given a textual explanation of code functionality, retrieve the corresponding code implementation.",
        TaskType.api_usage_retrieval: "Given a usage description of an API or library, retrieve code examples demonstrating the usage.",
        TaskType.bug_desc_retrieval: "Given a description of a software bug or unexpected behavior, retrieve relevant code that can help address the issue.",
        TaskType.pseudocode_retrieval: "Given a pseudocode description of an procedure, retrieve code implementations of the procedure.",
        TaskType.tutorial_query_retrieval: "Given a query related to a programming tutorial or learning material, retrieve code examples that are relevant to the query.",
        TaskType.algorithm_desc_retrieval: "Given a textual description of an algorithm, retrieve code implementations of the described algorithm.",
        
        # code2text
        TaskType.code_summary_retrieval: "Given a piece of code, retrieve the document string that summarizes the code.",
        TaskType.code_review_retrieval: "Given a piece of code, retrieve the review that explains its role.",
        TaskType.code_intent_retrieval: "Given a piece of code, retrieve the developer's intent or purpose described in a commit message or design document.",
        TaskType.code_optimization_retrieval: "Given a piece of code, retrieve optimization suggestions or performance analysis reports.",
        TaskType.tutorial_retrieval: "Given a piece of code, retrieve tutorials or how-to guides that demonstrate how to use or implement similar code.",
        TaskType.code_issue_discussion_retrieval: "Given a piece of code, retrieve discussions or issue reports related to the code, such as bug reports or feature requests.",
        TaskType.api_reference_retrieval: "Given a piece of code that uses specific APIs or libraries, retrieve the relevant API reference documentation for those APIs or libraries.",
        TaskType.code_walkthrough_retrieval: "Given a piece of code, retrieve a step-by-step walkthrough or detailed explanation of the code's logic and execution flow.",
        TaskType.code_error_explanation_retrieval: "Given a piece of code, retrieve the document that explains potential errors or exceptions that may arise from the code.",
        TaskType.code_to_requirement_retrieval: "Given a piece of code, retrieve the software requirement or user story it fulfills.",
        
        # code2code
        TaskType.code_context_retrieval: "Given a piece of code segment, retrieve the code segment that is the latter part of the code.",
        TaskType.similar_code_retrieval: "Given a piece of code, retrieve code that is semantically equivalent to the input code.",
        TaskType.code_translation_retrieval: "Given a piece of {src_language} code, retrieve {tgt_language} code that is semantically equivalent to the input code.",
        TaskType.code_refinement_retrieval: "Given a piece of code, retrieve a refined version of the code.",
        TaskType.secure_code_retrieval: "Given a piece of code, retrieve a version of the code with enhanced security measures or vulnerability fixes.",
        TaskType.code_version_update_retrieval: "Given a piece of code in an older language version, retrieve code updated to comply with the syntax or features of a newer language version.",
        TaskType.code_example_retrieval: "Given a code library or API, retrieve example code snippets that demonstrate how to use the library or API.",
        TaskType.code_dependency_retrieval: "Given a piece of code, retrieve all the code segments that the input code depends on, including libraries, functions, and variables.",
        TaskType.code_pattern_retrieval: "Given a piece of code, retrieve other code segments that follow the same design pattern or structure.",
        TaskType.code_history_retrieval: "Given a piece of code, retrieve previous versions or iterations of the code to understand its development history.",
        TaskType.code_integration_retrieval: "Given a piece of code, retrieve code that demonstrates how to integrate the input code with other systems or components.",
        TaskType.optimized_code_retrieval: "Given a piece of code, retrieve an optimized version of the code that improves performance, readability, or efficiency.",
        TaskType.code_simplification_retrieval: "Given a complex piece of code, retrieve a simplified version of the code that is easier to understand and maintain.",
        TaskType.code_modularization_retrieval: "Given a piece of code, retrieve a modularized version of the code that breaks it down into smaller, reusable components.",
        TaskType.code_augmentation_retrieval: "Given a piece of code, retrieve code that implements additional functionality while also preserving the original behavior.",
        TaskType.error_handling_code_retrieval: "Given a piece of code, retrieve code that incorporates error-checking or exception-handling mechanisms relevant to the input code.",
        TaskType.code_documentation_retrieval: "Given a piece of code, retrieve code with inline comments or documentation explaining its functionality.",
        TaskType.library_adaptation_retrieval: "Given a piece of code using one library or framework, retrieve code that achieves the same functionality using a different library or framework.",
        
        # hybrid
        TaskType.code_modification_retrieval: "Given a code snippet and a natural language description of desired modifications, retrieve relevant code that implements the requested modifications.",
        # TaskType.code_modification_retrieval: "Given a question that consists of a mix of text and code snippets, retrieve relevant code that answers the question.",
        # TaskType.single_turn_code_qa: "Given a question that consists of a mix of text and code snippets, retrieve relevant code that answer the question.",
        # TaskType.multi_turn_code_qa: "Given a multi-turn conversation history that consists of a mix of text and code snippets, retrieve relevant code that answer the question.",
        TaskType.code_bug_fix_example_retrieval: "Given a code snippet containing a bug and a natural language description of the bug or error, retrieve code snippets that demonstrate solutions or fixes for similar bugs or errors (the desired documents).",
        TaskType.code_refactoring_pattern_retrieval: "Given a code snippet that could be improved and a natural language description of desired refactoring goals or patterns, retrieve code snippets that exemplify similar refactoring techniques or patterns (the desired documents).",
        TaskType.code_style_guideline_example_retrieval: "Given a code snippet and a natural language query describing a desired coding style or best practice, retrieve code snippets that adhere to the specified style guidelines or best practices (the desired documents).",
        TaskType.code_migration_retrieval: "Given a code snippet and a natural language description of a specific migration requirement, retrieve code snippets that demonstrate how to migrate the code to meet the requirement.",
        TaskType.code_optimization_hybrid_retrieval: "Given a code snippet and a natural language request for specific optimization, retrieve relevant code that implements the requested optimization.",
        TaskType.code_comparison_retrieval: "Given two code snippets and a natural language query about their differences or similarities, retrieve relevant document that explains the differences or similarities between the two code snippets.",
        TaskType.code_best_practices_retrieval: "Given a code snippet and a natural language query about coding best practices, retrieve relevant document including guidelines, design patterns, or recommendations that can help improve the quality of the code.",
        TaskType.security_vulnerability_fix_retrieval: "Given a code snippet and a text description of a security concern, retrieve secure code alternatives that address the security vulnerability.",
    }
    
    task_type_to_main_type: Dict[TaskType, str] = {
        # text2code
        TaskType.web_code_retrieval: "text2code",
        TaskType.code_contest_retrieval: "text2code",
        TaskType.text2sql_retrieval: "text2code",
        TaskType.error_message_retrieval: "text2code",
        TaskType.code_explanation_retrieval: "text2code",
        TaskType.api_usage_retrieval: "text2code",
        TaskType.bug_desc_retrieval: "text2code",
        TaskType.pseudocode_retrieval: "text2code",
        TaskType.tutorial_query_retrieval: "text2code",
        TaskType.algorithm_desc_retrieval: "text2code",
        
        # code2text
        TaskType.code_summary_retrieval: "code2text",
        TaskType.code_review_retrieval: "code2text",
        TaskType.code_intent_retrieval: "code2text",
        TaskType.code_optimization_retrieval: "code2text",
        TaskType.tutorial_retrieval: "code2text",
        TaskType.code_issue_discussion_retrieval: "code2text",
        TaskType.api_reference_retrieval: "code2text",
        TaskType.code_walkthrough_retrieval: "code2text",
        TaskType.code_error_explanation_retrieval: "code2text",
        TaskType.code_to_requirement_retrieval: "code2text",
        
        # code2code
        TaskType.code_context_retrieval: "code2code",
        TaskType.similar_code_retrieval: "code2code",
        TaskType.code_translation_retrieval: "code2code",
        TaskType.code_refinement_retrieval: "code2code",
        TaskType.secure_code_retrieval: "code2code",
        TaskType.code_version_update_retrieval: "code2code",
        TaskType.code_example_retrieval: "code2code",
        TaskType.code_dependency_retrieval: "code2code",
        TaskType.code_pattern_retrieval: "code2code",
        TaskType.code_history_retrieval: "code2code",
        TaskType.code_integration_retrieval: "code2code",
        TaskType.optimized_code_retrieval: "code2code",
        TaskType.code_simplification_retrieval: "code2code",
        TaskType.code_modularization_retrieval: "code2code",
        TaskType.code_augmentation_retrieval: "code2code",
        TaskType.error_handling_code_retrieval: "code2code",
        TaskType.code_documentation_retrieval: "code2code",
        TaskType.library_adaptation_retrieval: "code2code",
        
        # hybrid
        TaskType.code_modification_retrieval: "hybrid",
        # TaskType.single_turn_code_qa: "hybrid",
        # TaskType.multi_turn_code_qa: "hybrid",
        TaskType.code_bug_fix_example_retrieval: "hybrid",
        TaskType.code_refactoring_pattern_retrieval: "hybrid",
        TaskType.code_style_guideline_example_retrieval: "hybrid",
        TaskType.code_migration_retrieval: "hybrid",
        TaskType.code_optimization_hybrid_retrieval: "hybrid",
        TaskType.code_comparison_retrieval: "hybrid",
        TaskType.code_best_practices_retrieval: "hybrid",
        TaskType.security_vulnerability_fix_retrieval: "hybrid",
    }

    if isinstance(task_type, str):
        task_type = TaskType[task_type]
    
    task_instruction = task_type_to_instruct[task_type]
    main_task_type = task_type_to_main_type[task_type]
    
    return main_task_type, task_type, task_instruction


class Language(Enum):
    # 主流语言 (2): 每种任务和每种 code language 均生产 (不包含文本的只生产 English)
    en = 'English'  # 英语
    zh = 'Simplified Chinese'  # 简体中文
    
    # 其他语言 (20)：从 text2code, code2text 中各 sample 3 个任务类型，再从 High 的 code language (java python javascript php ruby go csharp cplusplus) 中 sample 3 个 code language 出来，每个下面是 750 条，总共是 20 * 5 * 3 * 750 + 20 * 750 = 240K 条 (12K / language)
    ## Tasks: 1) web_code_retrieval, code_explanation_retrieval, text2sql_retrieval; 
    #         2) code_review_retrieval, code_walkthrough_retrieval, code_to_requirement_retrieval
    ## Code Languages: random sample 3 code languages
    ar = 'Arabic'  # 阿拉伯语
    bn = 'Bengali'  # 孟加拉语
    es = 'Spanish'  # 西班牙语
    fa = 'Persian'  # 波斯语
    fi = 'Finnish'  # 芬兰语
    fr = 'French'  # 法语
    hi = 'Hindi'  # 印地语
    id = 'Indonesian'  # 印度尼西亚语
    ja = 'Japanese'  # 日语
    ko = 'Korean'  # 韩语
    ru = 'Russian'  # 俄语
    sw = 'Swahili'  # 斯瓦希里语
    te = 'Telugu'  # 泰卢固语
    th = 'Thai'  # 泰语
    de = 'German'  # 德语
    yo = 'Yoruba'  # 约鲁巴语
    it = 'Italian'  # 意大利语
    pt = 'Portuguese'  # 葡萄牙语
    vi = 'Vietnamese'  # 越南语
    zh_tw = 'Traditional Chinese'   # 繁体中文

    # nl = 'Dutch'  # 荷兰语
    # no = 'Norwegian'  # 挪威语
    # sv = 'Swedish'  # 瑞典语
    # da = 'Danish'  # 丹麦语
    # pl = 'Polish'  # 波兰语
    # cs = 'Czech'  # 捷克语
    # hu = 'Hungarian'  # 匈牙利语
    # el = 'Greek'  # 希腊语
    # he = 'Hebrew'  # 希伯来语
    # tr = 'Turkish'  # 土耳其语
    # ku = 'Kurdish'  # 库尔德语
    # ur = 'Urdu'  # 乌尔都语
    # gu = 'Gujarati'  # 古吉拉特语
    # pa = 'Punjabi'  # 旁遮普语
    # ta = 'Tamil'  # 泰米尔语
    # kn = 'Kannada'  # 卡纳达语
    # ml = 'Malayalam'  # 马拉雅拉姆语
    # mr = 'Marathi'  # 马拉地语
    # ms = 'Malay'  # 马来语
    # my = 'Burmese'  # 缅甸语
    # jv = 'Javanese'  # 爪哇语
    # km = 'Khmer'  # 高棉语
    # yue = 'Cantonese'  # 粤语
    # zu = 'Zulu'  # 祖鲁语
    # ha = 'Hausa'  # 豪萨语
    # am = 'Amharic'  # 阿姆哈拉语
    # ig = 'Igbo'  # 伊博语
    # qu = 'Quechua'  # 克丘亚语
    # nah = 'Nahuatl'  # 纳瓦特尔语
    # ht = 'Haitian Creole'  # 海地克里奥尔语
    # tl = 'Tagalog'  # 塔加alog
    # mi = 'Maori'  # 毛利语
    # mn = 'Mongolian'  # 蒙古语

class CodeLanguage(Enum):
    # High (8): 3000 / language
    java = "Java"
    python = "Python"
    javascript = "JavaScript"
    php = "PHP"
    ruby = "Ruby"
    go = "GO"
    csharp = "C#"
    cplusplus = "C++"
    # Medium (6): 1500 / language
    c = "C"
    rust = "Rust"
    typescript = "TypeScript"
    perl = "Perl"
    shell = "Shell"
    sql = "SQL"
    # Low (6): 750 / language
    batchfile = "Batchfile"
    fortran = "FORTRAN"
    haskell = "Haskell"
    lua = "Lua"
    powershell = "PowerShell"
    visual_basic = "Visual Basic"
    
    # NULL for tasks that do not require code language
    null = ""
    
    # assembly = "Assembly"
    # cmake = "CMake"
    # css = "CSS"
    # dockerfile = "Dockerfile"
    # html = "HTML"
    # julia = "Julia"
    # makefile = "Makefile"
    # markdown = "Markdown"
    # scala = "Scala"
    # tex = "TeX"


# 16 * 2 = 32, 3000 per pair (32 * 3000 = 96000)
CODE_TRANSLATION_RETRIEVAL_PAIRS = [
    # c <-> cplusplus <-> csharp <-> java
    (CodeLanguage.c, CodeLanguage.cplusplus),
    (CodeLanguage.c, CodeLanguage.csharp),
    (CodeLanguage.c, CodeLanguage.java),
    (CodeLanguage.cplusplus, CodeLanguage.csharp),
    (CodeLanguage.cplusplus, CodeLanguage.java),
    (CodeLanguage.csharp, CodeLanguage.java),
    # python <-> ruby <-> perl
    (CodeLanguage.python, CodeLanguage.ruby),
    (CodeLanguage.python, CodeLanguage.perl),
    (CodeLanguage.ruby, CodeLanguage.perl),
    # javascript <-> typescript <-> php
    (CodeLanguage.javascript, CodeLanguage.typescript),
    (CodeLanguage.javascript, CodeLanguage.php),
    (CodeLanguage.typescript, CodeLanguage.php),
    # rust <-> go <-> cplusplus
    (CodeLanguage.rust, CodeLanguage.go),
    (CodeLanguage.rust, CodeLanguage.cplusplus),
    (CodeLanguage.go, CodeLanguage.cplusplus),
    # python <-> cplusplus
    (CodeLanguage.python, CodeLanguage.cplusplus),
]


@dataclass
class Task:
    task_type: TaskType
    language: Language
    code_language: CodeLanguage = CodeLanguage.null
    task_instruction: str = None
    tgt_code_language: CodeLanguage = CodeLanguage.null
    main_task_type: str = None


def get_task(
    task_type: str,
    language: str,
    code_language: str,
    tgt_code_language: Optional[str] = None
) -> Task:
    main_task_type, task_type, task_instruction = get_task_def_by_task_type(task_type)

    if tgt_code_language is None:
        tgt_code_language = "null"

    language = Language[language]
    code_language = CodeLanguage[code_language]
    tgt_code_language = CodeLanguage[tgt_code_language]

    task_instruction = task_instruction.replace("{src_language}", code_language.value).replace("{tgt_language}", tgt_code_language.value)

    task = Task(
        task_type=task_type,
        language=language,
        code_language=code_language,
        task_instruction=task_instruction,
        tgt_code_language=tgt_code_language,
        main_task_type=main_task_type
    )
    return task


SPECIAL_TASK_STEPS = {
    # TaskType.code_contest_retrieval: 2,
    TaskType.code_modification_retrieval: 2,
    TaskType.code_issue_discussion_retrieval: 2,
    TaskType.code_version_update_retrieval: 2,
    TaskType.code_bug_fix_example_retrieval: 2,
    TaskType.code_refactoring_pattern_retrieval: 2,
    TaskType.code_style_guideline_example_retrieval: 2,
    TaskType.bug_desc_retrieval: 2,
    TaskType.code_migration_retrieval: 2,
    TaskType.code_optimization_hybrid_retrieval: 2,
    TaskType.code_comparison_retrieval: 2,
    TaskType.code_best_practices_retrieval: 2,
    TaskType.security_vulnerability_fix_retrieval: 2,
}


def get_pos_as_input_by_task_type(task_type: TaskType) -> bool:
    """
    Get `pos_as_input` by task type.
    `pos_as_input=True` means that when generating a pair of query and pos, the pos is the input used for LLM generation. For example, text2code tasks: web_code_retrieval, code_contest_retrieval, text2sql_retrieval.
    `pos_as_input=False` means that when generating a pair of query and pos, the query is the input used for LLM generation. For example, code2text tasks: code_summary_retrieval, code_review_retrieval.
    """
    # TODO: Add more task types
    SPECIAL_TASKS = {
        # hybrid
        TaskType.code_bug_fix_example_retrieval: False,
        TaskType.code_refactoring_pattern_retrieval: False,
        TaskType.code_style_guideline_example_retrieval: False,
        TaskType.code_migration_retrieval: False,
        TaskType.code_optimization_hybrid_retrieval: False,
        TaskType.code_comparison_retrieval: False,
        TaskType.code_best_practices_retrieval: False,
        TaskType.security_vulnerability_fix_retrieval: False,
    }
    
    if task_type in SPECIAL_TASKS:
        return SPECIAL_TASKS[task_type]
    
    # normal rules
    main_task_type, _, _ = get_task_def_by_task_type(task_type)
    if main_task_type in ["text2code", "hybrid"]:
        return True
    elif main_task_type in ["code2text", "code2code"]:
        return False
    else:
        raise ValueError(f"Invalid task type: {task_type}")


def get_generation_prompt(
    task: Task,
    text: str,
    text_b: Optional[str] = None,
    examples: Optional[List[dict]] = None,
    idx: Optional[int] = None
) -> str:
    """
    Given a task, return the generation prompt for the task.
    
    Args:
    - task: Task: the task object
    - text: str: the input text
    - text_b: str: the second input text (optional), used for code_modification_retrieval task
    - examples: List[dict]: the examples for the task
    - idx: int: the index of gen_instruction in the instruction list (optional), used for tasks that need multiple steps to generate the output
    
    Returns:
    - gen_prompt: str: the generation prompt
    """
    
    task_to_gen_instruction: Dict[TaskType, str] = {
        # text2code (gen: code -> text)
        TaskType.web_code_retrieval: "Given a piece of {code_language} code, generate a web query in {language} that can be solved by the code.",
        TaskType.code_contest_retrieval: "Given a piece of {code_language} code, generate a code contest description in {language} that can be solved by the code.",
        TaskType.text2sql_retrieval: "Given a piece of {code_language} code, generate a text query in {language} for which the code is the appropriate response.",
        TaskType.error_message_retrieval: "Given a piece of {code_language} code, generate a possible error message in {language} that can be resolved by the code.",
        TaskType.code_explanation_retrieval: "Given a piece of {code_language} code, generate a textual explanation in {language} of the code functionality.",
        TaskType.api_usage_retrieval: "Given a piece of {code_language} code, generate a usage description of an API or library in {language} that can be demonstrated by the code as an example.",
        TaskType.bug_desc_retrieval: [
            "Given a piece of {code_language} code, modify some details of the code to introduce one or more bugs.",
            "Given a piece of {code_language} code with one or more bugs, generate a description of the bugs in {language}.",
        ],
        TaskType.pseudocode_retrieval: "Given a piece of {code_language} code, generate a pseudocode in {language} that describes the code functionality.",
        TaskType.tutorial_query_retrieval: "Given a piece of {code_language} code, generate a programming tutorial query in {language} that can be answered by the code as an example.",
        TaskType.algorithm_desc_retrieval: "Given a piece of {code_language} code, generate an algorithm description in {language} that can be implemented by the code.",
        
        # code2text (gen: code -> text)
        TaskType.code_summary_retrieval: "Given a piece of {code_language} code, generate a summary in {language} of the code.",
        TaskType.code_review_retrieval: "Given a piece of {code_language} code, generate a review in {language} that explains its role.",
        TaskType.code_intent_retrieval: "Given a piece of {code_language} code, generate a developer's intent or purpose described in a commit message or design document in {language}.",
        TaskType.code_optimization_retrieval: "Given a piece of {code_language} code, generate code optimization suggestions or performance analysis reports in {language}.",
        TaskType.tutorial_retrieval: "Given a piece of {code_language} code, generate tutorials or how-to guides that demonstrate how to use or implement similar code in {language}.",
        TaskType.code_issue_discussion_retrieval: [
            "Given a piece of {code_language} code, generate a version with some bugs.",
            "Given a piece of {code_language} code, generate a discussion of the code's issues or bugs in {language}, such as bug reports or feature requests.",
        ],
        TaskType.api_reference_retrieval: "Given a piece of {code_language} code, generate the relevant API reference documentation in {language} that can be used to understand the code.",
        TaskType.code_walkthrough_retrieval: "Given a piece of {code_language} code, generate a step-by-step walkthrough or detailed explanation of the code's logic and execution flow in {language}.",
        TaskType.code_error_explanation_retrieval: "Given a piece of {code_language} code, generate a detailed explanation of the errors or exceptions that may arise from the code in {language}.",
        TaskType.code_to_requirement_retrieval: "Given a piece of {code_language} code, generate a software requirement or user story it fulfills in {language}.",

        # code2code (gen: code-prefix -> code-suffix)
        TaskType.code_context_retrieval: "Given a piece of {code_language} code, generate a piece of code that is the latter part of the input code.",
        TaskType.similar_code_retrieval: "Given a piece of {code_language} code, generate a piece of {code_language} code that is semantically equivalent to the input code.",
        TaskType.code_translation_retrieval: "Given a piece of {code_language} code, generate a piece of {tgt_code_language} code that is semantically equivalent to the input code.",
        # src_language <-> code_language, tgt_language <-> tgt_code_language
        TaskType.code_refinement_retrieval: "Given a piece of {code_language} code, generate a refined version of the code.",
        TaskType.secure_code_retrieval: "Given a piece of {code_language} code, generate a a version of the code with enhanced security measures or vulnerability fixes.",
        TaskType.code_version_update_retrieval: [
            "Given a piece of {code_language} code, generate a lower-level version of the code.",
            "Given a piece of {code_language} code, update it with the syntax or features of a newer language version.",
        ],
        TaskType.code_example_retrieval: "Given a piece of {code_language} code, generate a piece of {code_language} code that is a good example of the code's usage.",
        TaskType.code_dependency_retrieval: "Given a piece of {code_language} code, generate the code segments that the input code depends on, including libraries, functions, and variables.",
        TaskType.code_pattern_retrieval: "Given a piece of {code_language} code, generate a piece of {code_language} code that follows the same design pattern or structure.",
        TaskType.code_history_retrieval: "Given a piece of {code_language} code, generate a piece of {code_language} code that is a historical version or iteration of the code.",
        TaskType.code_integration_retrieval: "Given a piece of {code_language} code, generate a piece of {code_language} code that integrates the input code with other systems or components.",
        TaskType.optimized_code_retrieval: "Given a piece of {code_language} code, generate an optimized version of the code that improves performance, readability, or efficiency.",
        TaskType.code_simplification_retrieval: "Given a piece of {code_language} code, generate a simplified version of the code that is easier to understand and maintain.",
        TaskType.code_modularization_retrieval: "Given a piece of {code_language} code, generate a modularized version of the code that breaks it down into smaller, reusable components.",
        TaskType.code_augmentation_retrieval: "Given a piece of {code_language} code, generate a piece of code that implements additional functionality while preserving the original behavior.",
        TaskType.error_handling_code_retrieval: "Given a piece of {code_language} code, generate a piece of code that incorporates error-checking or exception-handling mechanisms relevant to the input code.",
        TaskType.code_documentation_retrieval: "Given a piece of {code_language} code, generate a piece of code with inline comments or documentation explaining its functionality.",
        TaskType.library_adaptation_retrieval: "Given a piece of {code_language} code, generate a piece of code that achieves the same functionality using a different library or framework.",
        
        # hybrid (gen: code -> hybrid)
        TaskType.code_modification_retrieval: [
            "Given a piece of input code and a piece of output code, generate the differences in {language} between the input code and output code.",
            "Given the differences in {language} between a piece of input code and a piece of output code, generate a code modification instruction in {language} that uses only the information from the differences to transform the input code into the output code.",
        ],
        # TaskType.single_turn_code_qa: "Given a piece of code, generate a question that consists of a mix of {language} text and code snippets, and can be answered by the provided code.",
        # TaskType.multi_turn_code_qa: "Given a piece of code, generate a multi-turn conversation history that consists of a mix of {language} text and code snippets, and can be answered by the provided code.",
        TaskType.code_bug_fix_example_retrieval: [
            "Given a piece of {code_language} code, generate a buggy version of the code and a description in {language} of the bug or error.",
            "Given a piece of {code_language} code and a natural language description of the bug or error, generate a piece of {code_language} code that demonstrates a solution or fix for the bug or error.",
        ],
        TaskType.code_refactoring_pattern_retrieval: [
            "Given a piece of {code_language} code, generate a description of the desired refactoring goals or patterns in {language}.",
            "Given a piece of {code_language} code and a natural language description of the desired refactoring goals or patterns, generate a piece of {code_language} code that exemplifies similar refactoring techniques or patterns.",
        ],
        TaskType.code_style_guideline_example_retrieval: [
            "Given a piece of {code_language} code, generate a query describing a desired coding style or best practice to improve it in {language}.",
            "Given a piece of {code_language} code and a natural language query describing the desired style guidelines or best practices, generate a piece of {code_language} code that adheres to the specified style guidelines or best practices.",
        ],
        TaskType.code_migration_retrieval: [
            "Given a piece of {code_language} code, generate a specific migration requirement in {language} based on the code.",
            "Given a piece of {code_language} code and a natural language description of a specific migration requirement, generate a piece of {code_language} code that meets the migration requirement.",
        ],
        TaskType.code_optimization_hybrid_retrieval: [
            "Given a piece of {code_language} code, generate a question in {language} that requests a specific optimization for the code.",
            "Given a piece of {code_language} code and a natural language request in {language} for specific optimization, generate a piece of output code that implements the requested optimization.",
        ],
        TaskType.code_comparison_retrieval: [
            "Given a piece of input code and a piece of output code, generate a question in {language} about their differences or similarities.",
            "Given a piece of input code and a piece of output code, and a natural language question in {language} about their differences or similarities, generate a response that answer the question.",
        ],
        TaskType.code_best_practices_retrieval: [
            "Given a piece of {code_language} code, generate a question in {language} about coding best practices related to the code.",
            "Given a piece of {code_language} code and a natural language question in {language} about coding best practices related to the code, generate a response including guidelines, design patterns, or recommendations that can help improve the quality of the code.",
        ],
        TaskType.security_vulnerability_fix_retrieval: [
            "Given a piece of {code_language} code, generate a text description in {language} of a possible security concern in the code.",
            "Given a piece of {code_language} code and a text description in {language} of a security concern, generate secure code alternatives that address the vulnerability.",
        ],
    }
    
    task_to_gen_output: Dict[TaskType, str] = {
        # text2code (gen: code -> text)
        TaskType.web_code_retrieval: "the generated web query in {language}",
        TaskType.code_contest_retrieval: "the generated code contest description in {language}",
        TaskType.text2sql_retrieval: "the generated text query in {language}",
        TaskType.error_message_retrieval: "the generated error message in {language}",
        TaskType.code_explanation_retrieval: "the generated explanation in {language}",
        TaskType.api_usage_retrieval: "the generated API or library usage description in {language}",
        TaskType.bug_desc_retrieval: [
            "the modified code with one or more bugs",
            "the generated bug description in {language}",
        ],
        TaskType.pseudocode_retrieval: "the generated pseudocode in {language}",
        TaskType.tutorial_query_retrieval: "the generated programming tutorial query in {language}",
        TaskType.algorithm_desc_retrieval: "the generated algorithm description in {language}",
        
        # code2text (gen: code -> text)
        TaskType.code_summary_retrieval: "the generated summary in {language}",
        TaskType.code_review_retrieval: "the generated review in {language}",
        TaskType.code_intent_retrieval: "the generated intent in {language}",
        TaskType.code_optimization_retrieval: "the generated optimization suggestions or performance analysis reports in {language}",
        TaskType.tutorial_retrieval: "the generated tutorial in {language}",
        TaskType.code_issue_discussion_retrieval: [
            "the generated buggy code",
            "the generated error explanation in {language}",
        ],
        TaskType.api_reference_retrieval: "the generated API reference documentation in {language}",
        TaskType.code_walkthrough_retrieval: "the generated walkthrough in {language}",
        TaskType.code_error_explanation_retrieval: "the generated error explanation in {language}",
        TaskType.code_to_requirement_retrieval: "the generated requirement in {language}",

        # code2code (gen: code-prefix -> code-suffix)
        TaskType.code_context_retrieval: "the generated piece of {code_language} code",
        TaskType.similar_code_retrieval: "the generated piece of {code_language} code",
        TaskType.code_translation_retrieval: "the generated piece of {tgt_code_language} code",
        TaskType.code_refinement_retrieval: "the generated piece of {code_language} code",
        TaskType.secure_code_retrieval: "the generated piece of {code_language} code",
        TaskType.code_version_update_retrieval: [
            "the generated piece of {code_language} code",
            "the generated piece of {code_language} code",
        ],
        TaskType.code_example_retrieval: "the generated piece of {code_language} code",
        TaskType.code_dependency_retrieval: "the generated piece of {code_language} code",
        TaskType.code_pattern_retrieval: "the generated piece of {code_language} code",
        TaskType.code_history_retrieval: "the generated piece of {code_language} code",
        TaskType.code_integration_retrieval: "the generated piece of {code_language} code",
        TaskType.optimized_code_retrieval: "the generated piece of {code_language} code",
        TaskType.code_simplification_retrieval: "the generated piece of {code_language} code",
        TaskType.code_modularization_retrieval: "the generated piece of {code_language} code",
        TaskType.code_modification_retrieval: "the generated piece of {code_language} code",
        TaskType.code_augmentation_retrieval: "the generated piece of {code_language} code",
        TaskType.error_handling_code_retrieval: "the generated piece of {code_language} code",
        TaskType.code_documentation_retrieval: "the generated piece of {code_language} code",
        TaskType.library_adaptation_retrieval: "the generated piece of {code_language} code",
        
        # hybrid (gen: code -> hybrid)
        TaskType.code_modification_retrieval: [
            "the generated differences in {language} between the input code and output code",
            "the generated modification instruction in {language}",
        ],
        # TaskType.single_turn_code_qa: "the generated question that consists of a mix of {language} text and code snippets",
        # TaskType.multi_turn_code_qa: "the generated multi-turn conversation history that consists of a mix of {language} text and code snippets",
        TaskType.code_bug_fix_example_retrieval: [
            "the generated buggy version of the code and a description in {language} of the bug or error",
            "the generated piece of {code_language} code"
        ],
        TaskType.code_refactoring_pattern_retrieval: [
            "the generated description of the desired refactoring goals or patterns in {language}",
            "the generated piece of {code_language} code"
        ],
        TaskType.code_style_guideline_example_retrieval: [
            "the generated query describing a desired coding style or best practice to improve it in {language}",
            "the generated piece of {code_language} code"
        ],
        TaskType.code_migration_retrieval: [
            "the generated specific migration requirement in {language} based on the code",
            "the generated piece of {code_language} code"
        ],
        TaskType.code_optimization_hybrid_retrieval: [
            "the generated question in {language} that requests a specific optimization for the code",
            "the generated piece of {code_language} code",
        ],
        TaskType.code_comparison_retrieval: [
            "the generated question in {language} about their differences or similarities",
            "the generated response in {language}",
        ],
        TaskType.code_best_practices_retrieval: [
            "the generated question in {language} about coding best practices related to the code",
            "the generated response in {language}",
        ],
        TaskType.security_vulnerability_fix_retrieval: [
            "the generated text description in {language} of a possible security concern in the code",
            "the generated piece of {code_language} code",
        ],
    }
    
    gen_instruction = task_to_gen_instruction[task.task_type]
    gen_output = task_to_gen_output[task.task_type]
    
    if idx is not None:
        assert isinstance(gen_instruction, list)
        gen_instruction = gen_instruction[idx]
        assert isinstance(gen_output, list)
        gen_output = gen_output[idx]
    
    assert isinstance(gen_instruction, str)
    assert isinstance(gen_output, str)
    
    gen_instruction = gen_instruction.replace("{language}", task.language.value).replace("{code_language}", task.code_language.value).replace("{tgt_code_language}", task.tgt_code_language.value)
    gen_output = gen_output.replace("{language}", task.language.value).replace("{code_language}", task.code_language.value).replace("{tgt_code_language}", task.tgt_code_language.value)
    
    if task.task_type == TaskType.code_modification_retrieval:
        if idx == 0:
            assert text_b is not None
            gen_prompt = f"""\
{gen_instruction}

Input code:
```{task.code_language.name}
{text}
```

Output code:
```{task.code_language.name}
{text_b}
```

Note:
- Your output must always be a string, only containing {gen_output}.
- Your output should be independent of the given code, which means that it should not contain the pronouns such as "it", "this", "that", "the given", "the provided", etc.

Remember do not explain your output or output anything else. Your output:"""
            return gen_prompt
        elif idx == 1:
            prefix = "Differences:"
        else:
            raise ValueError("Invalid idx for code_modification_retrieval task")
    elif task.task_type == TaskType.code_comparison_retrieval:
        if idx == 0:
            assert text_b is not None
            gen_prompt = f"""\
{gen_instruction}

Input code:
```{task.code_language.name}
{text}
```

Output code:
```{task.code_language.name}
{text_b}
```

Note:
- Your output must always be a string, only containing {gen_output}.
- Your output should be independent of the given code, which means that it should not contain the pronouns such as "it", "this", "that", "the given", "the provided", etc.

Remember do not explain your output or output anything else. Your output:"""
            return gen_prompt
        elif idx == 1:
            prefix = "Hybrid:"
        else:
            raise ValueError("Invalid idx for code_comparison_retrieval task")
    elif task.task_type in [
        TaskType.code_bug_fix_example_retrieval,
        TaskType.code_refactoring_pattern_retrieval,
        TaskType.code_style_guideline_example_retrieval,
        TaskType.code_migration_retrieval,
        TaskType.code_optimization_hybrid_retrieval,
        TaskType.code_best_practices_retrieval,
        TaskType.security_vulnerability_fix_retrieval,
    ]:
        if idx == 0:
            prefix = "Code:"
        elif idx == 1:
            prefix = "Hybrid:"
        else:
            raise ValueError("Invalid idx for hybrid task")
    else:
        prefix = "Code:"

    gen_prompt = f"""\
{gen_instruction}

{prefix}
```{task.code_language.name}
{text}
```

Note:
- Your output must always be a string, only containing {gen_output}.
- Your output should be independent of the given code, which means that it should not contain the pronouns such as "it", "this", "that", "the given", "the provided", etc.

"""

    if idx != 0 and examples is not None:
        examples_str_list = [f"""\
- Example {i + 1}:
    {prefix}
    ```{task.code_language.name}
    {example['input']}
    ```
    Expected Output ({gen_output}):
    ```
    {example['output']}
    ```

""" for i, example in enumerate(examples)]
        
        gen_prompt += f"""\
Here are a few examples for your reference:
{''.join(examples_str_list)}
"""

    gen_prompt += "Remember do not explain your output or output anything else. Your output:"

    return gen_prompt


def get_quality_control_prompt(
    task: Task,
    query: str,
    pos: str,
) -> str:
    """
    Given a task, return the quality control prompt for the task.
    
    Args:
    - task: Task: the task object
    
    Returns:
    - qc_prompt: str: the quality control prompt
    """
    
    # return tuples of (mission, query_type, doc_type, qc_options)
    task_to_qc_mission: Dict[TaskType, str] = {
        # text2code
        TaskType.web_code_retrieval: (
            "judge whether the code can help answer the web search query",
            "the web search query",
            "the code",
            [
                "Yes, the code can help answer the web search query.",
                "No, the code cannot help answer the web search query.",
            ]
        ),
        TaskType.code_contest_retrieval: (
            "judge whether the code can help solve the code contest problem",
            "the code contest problem",
            "the code",
            [
                "Yes, the code can help solve the code contest problem.",
                "No, the code cannot help solve the code contest problem.",
            ]
        ),
        TaskType.text2sql_retrieval: (
            "judge whether the code is an appropriate response to the text query",
            "the text query",
            "the code",
            [
                "Yes, the code is an appropriate response to the text query.",
                "No, the code is not an appropriate response to the text query.",
            ]
        ),
        TaskType.error_message_retrieval: (
            "judge whether the code can help resolve the error message",
            "the error message",
            "the code",
            [
                "Yes, the code can help resolve the error message.",
                "No, the code cannot help resolve the error message.",
            ]
        ),
        TaskType.code_explanation_retrieval: (
            "judge whether the code implements the functionality described in the explanation",
            "the explanation",
            "the code",
            [
                "Yes, the code implements the functionality described in the explanation.",
                "No, the code does not implement the functionality described in the explanation.",
            ]
        ),
        TaskType.api_usage_retrieval: (
            "judge whether the code demonstrates the usage description of the API or library",
            "the API or library usage description",
            "the code",
            [
                "Yes, and the code demonstrates the usage description of the API or library.",
                "No, the code does not demonstrate the usage description of the API or library.",
            ]
        ),
        TaskType.bug_desc_retrieval: (
            "judge whether the code can help address the described bug",
            "the bug description",
            "the code",
            [
                "Yes, the code can help address the described bug.",
                "No, the code cannot help address the described bug.",
            ]
        ),
        TaskType.pseudocode_retrieval: (
            "judge whether the code implements the procedure described in the pseudocode",
            "the pseudocode",
            "the code",
            [
                "Yes, the code implements the procedure described in the pseudocode.",
                "No, the code does not implement the procedure described in the pseudocode.",
            ]
        ),
        TaskType.tutorial_query_retrieval: (
            "judge whether the code can answer the programming tutorial query",
            "the programming tutorial query",
            "the code",
            [
                "Yes, the code can answer the programming tutorial query.",
                "No, the code cannot answer the programming tutorial query.",
            ]
        ),
        TaskType.algorithm_desc_retrieval: (
            "judge whether the code implements the algorithm described in the text",
            "the algorithm description",
            "the code",
            [
                "Yes, the code implements the algorithm described in the text.",
                "No, the code does not implement the algorithm described in the text.",
            ]
        ),
        
        # code2text
        TaskType.code_summary_retrieval: (
            "judge whether the text summarizes the code",
            "the code",
            "the text",
            [
                "Yes, the text summarizes the code.",
                "No, the text does not summarize the code.",
            ]
        ),
        TaskType.code_review_retrieval: (
            "judge whether the review explains the role of the code",
            "the code",
            "the review",
            [
                "Yes, the review explains the role of the code.",
                "No, the review does not explain the role of the code.",
            ]
        ),
        TaskType.code_intent_retrieval: (
            "judge whether the text describes the intent of the code",
            "the code",
            "the text",
            [
                "Yes, the text describes the intent of the code.",
                "No, the text does not describe the intent of the code.",
            ]
        ),
        TaskType.code_optimization_retrieval: (
            "judge whether the text provides optimization suggestions or performance analysis reports for the code",
            "the code",
            "the text",
            [
                "Yes, the text provides optimization suggestions or performance analysis reports for the code.",
                "No, the text provides neither optimization suggestions nor performance analysis reports for the code.",
            ]
        ),
        TaskType.tutorial_retrieval: (
            "judge whether the text is a tutorial or how-to guide that demonstrates how to use or implement similar code",
            "the code",
            "the text",
            [
                "Yes, the text is a tutorial or how-to guide that demonstrates how to use or implement similar code.",
                "No, the text neither provides instructional guidance for using similar code nor demonstrates how to implement similar code.",
            ]
        ),
        TaskType.code_error_explanation_retrieval: (
            "judge whether the text describes potential errors or exceptions that may arise from the code",
            "the code",
            "the text",
            [
                "Yes, the text describes potential errors or exceptions that may arise from the code.",
                "No, the text neither describes potential errors nor discuss exceptions that may arise from the code.",
            ]
        ),
        TaskType.code_issue_discussion_retrieval: (
            "judge whether the text is a discussion or issue report related to the code",
            "the code",
            "the text",
            [
                "Yes, the text is a discussion or issue report related to the code.",
                "No, the text is neither a discussion about the code nor an issue report related to the code.",
            ]
        ),
        TaskType.api_reference_retrieval: (
            "judge whether the text is an API reference documentation for the APIs or libraries used in the code",
            "the code",
            "the text",
            [
                "Yes, the text is an API reference documentation for the APIs or libraries used in the code.",
                "No, the text is not an API reference documentation for the APIs or libraries used in the code.",
            ]
        ),
        TaskType.code_walkthrough_retrieval: (
            "judge whether the text is a step-by-step walkthrough or detailed explanation of the code's logic and execution flow",
            "the code",
            "the text",
            [
                "Yes, the text is a step-by-step walkthrough or detailed explanation of the code's logic and execution flow.",
                "No, the text is neither a step-by-step walkthrough nor a detailed explanation of the code's logic and execution flow.",
            ]
        ),
        TaskType.code_to_requirement_retrieval: (
            "judge whether the text is a software requirement or user story that the code fulfills",
            "the code",
            "the text",
            [
                "Yes, the text is a software requirement or user story that the code fulfills.",
                "No, the text is neither a software requirement nor a user story that the code fulfills.",
            ]
        ),
        
        # code2code
        TaskType.code_context_retrieval: (
            "judge whether the output code is the latter part of the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is the latter part of the input code.",
                "No, the output code is not the latter part of the input code.",
            ]
        ),
        TaskType.similar_code_retrieval: (
            "judge whether the output code is semantically equivalent to the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is semantically equivalent to the input code.",
                "No, the output code is not semantically equivalent to the input code.",
            ]
        ),
        TaskType.code_translation_retrieval: (
            "judge whether the output code is semantically equivalent to the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is semantically equivalent to the input code.",
                "No, the output code is not semantically equivalent to the input code.",
            ]
        ),
        TaskType.code_refinement_retrieval: (
            "judge whether the output code is a refined version of the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is a refined version of the input code.",
                "No, the output code is not a refined version of the input code.",
            ]
        ),
        TaskType.secure_code_retrieval: (
            "judge whether the output code is the version with enhanced security measures or vulnerability fixes compared to the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is the version with enhanced security measures or vulnerability fixes compared to the input code.",
                "No, the output code neither introduces security enhancements nor fixes vulnerabilities compared to the input code.",
            ]
        ),
        TaskType.code_version_update_retrieval: (
            "judge whether the output code is the version updated to comply with the syntax or features of a newer language version compared to the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is the version updated to comply with the syntax or features of a newer code language version compared to the input code.",
                "No, the output code neither adopts syntax updates nor introduces newer code language features compared to the input code.",
            ]
        ),
        TaskType.code_example_retrieval: (
            "judge whether the output code is the example code snippets that demonstrate how to use the library or API in the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is the example code snippets that demonstrate how to use the library or API in the input code.",
                "No, the output code is not the example code snippets that demonstrate how to use the library or API in the input code.",
            ]
        ),
        TaskType.code_dependency_retrieval: (
            "judge whether the output code is the code segments that the input code depends on, including libraries, functions, and variables.",
            "the input code",
            "the output code",
            [
                "Yes, the output code is the code segments that the input code depends on, including libraries, functions, and variables.",
                "No, the output code is not the code segments that the input code depends on, including libraries, functions, and variables.",
            ]
        ),
        TaskType.code_pattern_retrieval: (
            "judge whether the output code follows the same design pattern or structure as the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code follows the same design pattern or structure as the input code.",
                "No, the output code neither follows the same design pattern nor retains the same structure as the input code.",
            ]
        ),
        TaskType.code_history_retrieval: (
            "judge whether the output code is the historical version or iteration of the input code, and can help understand its development history.",
            "the input code",
            "the output code",
            [
                "Yes, the output code is the historical version or iteration of the input code, and can help understand its development history.",
                "No, the output code is not the historical version or iteration of the input code, and cannot help understand its development history.",
            ]
        ),
        TaskType.code_integration_retrieval: (
            "judge whether the output code demonstrates how to integrate the input code with other systems or components.",
            "the input code",
            "the output code",
            [
                "Yes, the output code demonstrates how to integrate the input code with other systems or components.",
                "No, the output code does not demonstrate how to integrate the input code with other systems or components.",
            ]
        ),
        TaskType.optimized_code_retrieval: (
            "judge whether the output code is an optimized version of the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is an optimized version of the input code.",
                "No, the output code is not an optimized version of the input code.",
            ]
        ),
        TaskType.code_simplification_retrieval: (
            "judge whether the output code is a simplified version of the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is a simplified version of the input code.",
                "No, the output code is not a simplified version of the input code.",
            ]
        ),
        TaskType.code_modularization_retrieval: (
            "judge whether the output code is a modularized version of the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code is a modularized version of the input code.",
                "No, the output code is not a modularized version of the input code.",
            ]
        ),
        TaskType.code_augmentation_retrieval: (
            "judge whether the output code implements additional functionality while preserving the original behavior of the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code implements additional functionality while preserving the original behavior of the input code.",
                "No, the output code does not implement additional functionality while preserving the original behavior of the input code.",
            ]
        ),
        TaskType.error_handling_code_retrieval: (
            "judge whether the output code incorporates error-checking or exception-handling mechanisms relevant to the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code incorporates error-checking or exception-handling mechanisms relevant to the input code.",
                "No, the output code does not incorporate error-checking or exception-handling mechanisms relevant to the input code.",
            ]
        ),
        TaskType.code_documentation_retrieval: (
            "judge whether the output code contains inline comments or documentation explaining the functionality of the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code contains inline comments or documentation explaining the functionality of the input code.",
                "No, the output code does not contain inline comments or documentation explaining the functionality of the input code.",
            ]
        ),
        TaskType.library_adaptation_retrieval: (
            "judge whether the output code achieves the same functionality using a different library or framework as the input code",
            "the input code",
            "the output code",
            [
                "Yes, the output code achieves the same functionality using a different library or framework as the input code.",
                "No, the output code does not achieve the same functionality using a different library or framework as the input code.",
            ]
        ),
        
        # hybrid
        TaskType.code_modification_retrieval: (
            "judge whether the output code implements the requested modification described in the query",
            "the query",
            "the output code",
            [
                "Yes, the output code implements the requested modification described in the query.",
                "No, the output code does not implement the requested modification described in the query.",
            ]
        ),
        # TaskType.single_turn_code_qa: "judge whether the output code can answer the question",
        # TaskType.multi_turn_code_qa: "judge whether the output code can answer the question",
        TaskType.code_bug_fix_example_retrieval: (
            "judge whether the output code fixes the bug or error described in the query.",
            "the query",
            "the output code",
            [
                "Yes, the output code fixes the bug or error described in the query.",
                "No, the output code does not fix the bug or error described in the query.",
            ]
        ),
        TaskType.code_refactoring_pattern_retrieval: (
            "judge whether the output code exemplifies similar refactoring techniques or patterns described in the query",
            "the query",
            "the output code",
            [
                "Yes, the output code exemplifies similar refactoring techniques or patterns described in the query.",
                "No, the output code does not exemplify similar refactoring techniques or patterns described in the query.",
            ]
        ),
        TaskType.code_style_guideline_example_retrieval: (
            "judge whether the output code adheres to the specified style guidelines or best practices described in the query",
            "the query",
            "the output code",
            [
                "Yes, the output code adheres to the specified style guidelines or best practices described in the query.",
                "No, the output code does not adhere to the specified style guidelines or best practices described in the query.",
            ]
        ),
        TaskType.code_migration_retrieval: (
            "judge whether the output code meets the migration requirement described in the query",
            "the query",
            "the output code",
            [
                "Yes, the output code meets the migration requirement described in the query.",
                "No, the output code does not meet the migration requirement described in the query.",
            ]
        ),
        TaskType.code_optimization_hybrid_retrieval: (
            "judge whether the output code implements the requested optimization described in the query",
            "the query",
            "the output code",
            [
                "Yes, the output code implements the requested optimization described in the query.",
                "No, the output code does not implement the requested optimization described in the query.",
            ]
        ),
        TaskType.code_comparison_retrieval: (
            "judge whether the response can answer the question described in the query",
            "the query",
            "the response",
            [
                "Yes, the response can answer the question described in the query.",
                "No, the response cannot answer the question described in the query.",
            ]
        ),
        TaskType.code_best_practices_retrieval: (
            "judge whether the response can answer the question described in the query",
            "the query",
            "the response",
            [
                "Yes, the response can answer the question described in the query.",
                "No, the response cannot answer the question described in the query.",
            ]
        ),
        TaskType.security_vulnerability_fix_retrieval: (
            "judge whether the output code addresses the security vulnerability described in the query",
            "the query",
            "the output code",
            [
                "Yes, the output code addresses the security vulnerability described in the query.",
                "No, the output code does not address the security vulnerability described in the query.",
            ]
        ),
    }
    
    if task.main_task_type == "text2code":
        type_check_option = "the query contains code snippets or the document contains non-code content (plain text)."
    elif task.main_task_type == "code2text":
        type_check_option = "the query contains non-code content (plain text) or the document contains code snippets."
    elif task.main_task_type == "code2code":
        type_check_option = "either the query or the document contains non-code content (plain text)."
    else:
        type_check_option = "neither the query nor the document contains the mixed content of code and text content."
    
    qc_mission, query_type, doc_type, qc_options = task_to_qc_mission[task.task_type]
    
    pos_option = qc_options[0]
    neg_option = qc_options[1]
    
    # Init prompt
    # 0 代表 query / document 不符合 main task type
    # 1 代表 query / document 符合 main task type，且 judgment 是 positive
    # 2 代表 query / document 符合 main task type，且 judgment 是 negative
    # 输出中包含 1 即保留该 data
    qc_prompt = f"""\
Given a code retrieval task (Task), a query (Query), and a document (Document), your mission is to {qc_mission}.

Task ({task.main_task_type}): {task.task_instruction}

Query ({query_type}):
```
{query}
```

Document ({doc_type}):
```
{pos}
```

Your output must be one of the following options:
- 0: The query or document does not match the main task type ({task.main_task_type}), which means that {type_check_option}
- 1: The query and document match the main task type ({task.main_task_type}). The judgment is: {pos_option}
- 2: The query and document match the main task type ({task.main_task_type}). The judgment is: {neg_option}

Do not explain your answer in the output. Your output must be a single number (0 or 1 or 2).

Your output:"""
    
    return qc_prompt


class DocLength(Enum):
    len_0_500 = "_len-0-500.jsonl"
    len_500_1000 = "_len-500-1000.jsonl"
    len_1000_2000 = "_len-1000-2000.jsonl"
    len_2000_4000 = "_len-2000-4000.jsonl"
    len_4000_8000 = "_len-4000-8000.jsonl"
    len_8000_16000 = "_len-8000-16000.jsonl"
    len_16000_32000 = "_len-16000-32000.jsonl"


# only used for paper cmp: gen hard negative v.s. mine hard negative
def get_gen_hard_neg_prompt(task: Task, query: str, pos: str) -> str:
    """
    Given a task, return the generation hard negative prompt for the task.
    
    Args:
    - task: Task: the task object

    Returns:
    - gen_hard_neg_prompt: str: the generation hard negative prompt
    """
    gen_hard_neg_prompt = f"""\
Given a code retrieval task (Task), a query (Query), and a positive document (Positive Document), your mission is to generate a hard negative document that only appears relevant to the query under the code retrieval task.

Task ({task.main_task_type}): {task.task_instruction}

Query:
```
{query}
```

Positive Document:
```
{pos}
```

Note:
- Your output must always be a string, only containing the hard negative document.
- The hard negative document should be similar to the positive document in terms of content. If the positive document is a code snippet, the hard negative document should also be a code snippet. If the positive document is a text description, the hard negative document should also be a text description.

Remember do not explain your output or output anything else. Your output:"""

    return gen_hard_neg_prompt


NUM_HARD_NEGATIVES = 7
