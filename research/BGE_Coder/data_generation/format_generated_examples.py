import os
import json
from constant import Language, CodeLanguage, TaskType, CODE_TRANSLATION_RETRIEVAL_PAIRS, \
    get_pos_as_input_by_task_type


def format_generated_examples(
    file_path: str,
    save_path: str,
    task_type: TaskType
):
    if os.path.exists(save_path):
        return
    
    if not os.path.exists(file_path):
        print("====================================")
        print("Warning: file not found! Maybe need to generate it first.")
        print(f"file_path: {file_path}")
        return
    
    pos_as_input = get_pos_as_input_by_task_type(task_type)
    
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            
            if pos_as_input:
                _input = data["pos"][0]
                _output = data["query"]
            else:
                _input = data["query"]
                _output = data["pos"][0]

            if 'provided' in _input:
                continue
            if len(_input) > 12000 or len(_output) > 12000:
                continue

            data_list.append({
                "input": _input,
                "output": _output
            })
    
    if len(data_list) == 0:
        print("====================================")
        print("Warning: no data found!")
        print(f"file_path: {file_path}")
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)


def main():
    original_gen_examples_dir = "./examples"
    
    formatted_examples_dir = "./filtered_for_generation"
    
    for language in Language:
        for task_type in TaskType:
            if task_type == TaskType.code_translation_retrieval:
                for code_language_pair in CODE_TRANSLATION_RETRIEVAL_PAIRS:
                    code_language, tgt_code_language = code_language_pair
                    
                    file_path = os.path.join(
                        original_gen_examples_dir,
                        language.name, task_type.name, f"{language.name}-{code_language.name}-to-{tgt_code_language.name}-triplets.jsonl"
                    )
                    save_path = os.path.join(
                        formatted_examples_dir,
                        language.name, task_type.name, f"{code_language.name}-to-{tgt_code_language.name}_sample_examples.json"
                    )
                    
                    format_generated_examples(file_path, save_path, task_type)
                    
                for code_language_pair in CODE_TRANSLATION_RETRIEVAL_PAIRS:
                    tgt_code_language, code_language = code_language_pair
                    
                    file_path = os.path.join(
                        original_gen_examples_dir,
                        language.name, task_type.name, f"{language.name}-{code_language.name}-to-{tgt_code_language.name}-triplets.jsonl"
                    )
                    save_path = os.path.join(
                        formatted_examples_dir,
                        language.name, task_type.name, f"{code_language.name}-to-{tgt_code_language.name}_sample_examples.json"
                    )
                    
                    format_generated_examples(file_path, save_path, task_type)
                
            elif task_type == TaskType.text2sql_retrieval:
                file_path = os.path.join(
                    original_gen_examples_dir,
                    language.name, task_type.name, f"{language.name}-sql-triplets.jsonl"
                )
                save_path = os.path.join(
                    formatted_examples_dir,
                    language.name, task_type.name, "sql_sample_examples.json"
                )
                
                format_generated_examples(file_path, save_path, task_type)
            
            elif task_type == TaskType.code_context_retrieval:
                continue
            
            else:
                for code_language in CodeLanguage:
                    if code_language == CodeLanguage.null:
                        continue
                    
                    file_path = os.path.join(
                        original_gen_examples_dir,
                        language.name, task_type.name, f"{language.name}-{code_language.name}-triplets.jsonl"
                    )
                    save_path = os.path.join(
                        formatted_examples_dir,
                        language.name, task_type.name, f"{code_language.name}_sample_examples.json"
                    )
                    
                    format_generated_examples(file_path, save_path, task_type)
    
    print("All done!")


if __name__ == "__main__":
    main()
