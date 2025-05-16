import os
import json
import random
from tqdm import tqdm
from hashlib import md5
from warnings import warn
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from llm import LLM
from utils import clean_content
from constant import TaskType, Task, SPECIAL_TASK_STEPS, \
    get_task, get_generation_prompt, get_quality_control_prompt, \
    get_gen_hard_neg_prompt


def compute_md5(text: str):
    return md5(text.encode()).hexdigest()


class TripletGenerator(LLM):
    def __init__(
        self,
        model: str = "Qwen2-5-Coder-32B-Instruct",
        model_type: str = "open-source",
        port: int = 8000,
        cache_dir: Optional[str] = None
    ):
        super().__init__(model, model_type, port)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _gen_for_code_modification_retrieval(
        self,
        task: Task,
        text: str,
        text_b: Optional[str] = None,
        examples: Optional[List[dict]] = None,
        debug_mode: bool = False,
        **kwargs
    ):
        gen_prompt = get_generation_prompt(
            task=task,
            text=text,
            text_b=text_b,
            examples=examples,
            idx=0
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        diff = clean_content(response)
        gen_prompt = get_generation_prompt(
            task=task,
            text=diff,
            examples=examples,
            idx=1
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        modification_instr = clean_content(response)
        
        query = f"{modification_instr}\n```\n{text}\n```"
        pos = text_b
        
        if debug_mode:
            result = {
                "generation_prompt": gen_prompt,
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        else:
            result = {
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        return result
    
    def _gen_for_code_comparison_retrieval(
        self,
        task: Task,
        text: str,
        text_b: Optional[str] = None,
        examples: Optional[List[dict]] = None,
        debug_mode: bool = False,
        **kwargs
    ):
        gen_prompt = get_generation_prompt(
            task=task,
            text=text,
            text_b=text_b,
            examples=examples,
            idx=0
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        diff_question = clean_content(response)
        query = f"{diff_question}\n\nInput Code:\n```\n{text}\n```\n\nOutput Code:\n```\n{text_b}\n```"
        gen_prompt = get_generation_prompt(
            task=task,
            text=query,
            examples=examples,
            idx=1
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        pos = clean_content(response)

        if debug_mode:
            result = {
                "generation_prompt": gen_prompt,
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        else:
            result = {
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        return result
    
    def _gen_for_code_context_retrieval(
        self,
        task: Task,
        text: str,
        anchor_points: Optional[tuple] = (0.4, 0.7),
        **kwargs
    ):
        former_part, latter_part = self.split_text(
            text,
            anchor_points=anchor_points
        )
        result = {
            "prompt": task.task_instruction,
            "query": former_part,
            "pos": [latter_part],
            "neg": []
        }
        return result
    
    @staticmethod
    def _arrange_query_and_pos(task: Task, input_text: str, response: str):
        """
        Arrange the query and positive example based on the task type.
        
        Args:
        - task: Task
        - input_text: str
        - response: str
        
        Returns:
        - query: str
        - pos: str
        """
        # TODO: support more task types, including some special task types.
        if task.main_task_type in ["text2code", "hybrid"]:
            query = clean_content(response)
            pos = input_text
        else:
            query = input_text
            pos = clean_content(response)
        return query, pos
    
    def _gen_for_normal_task(
        self,
        task: Task,
        text: str,
        examples: Optional[List[dict]] = None,
        debug_mode: bool = False,
        **kwargs
    ):
        gen_prompt = get_generation_prompt(
            task=task,
            text=text,
            examples=examples
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        
        # Arrange the query and positive example based on the task type.
        query, pos = self._arrange_query_and_pos(
            task=task,
            input_text=text,
            response=response
        )
        
        if debug_mode:
            result = {
                "generation_prompt": gen_prompt,
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": [],
                "response": response
            }
        else:
            result = {
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        return result
    
    def _gen_for_bug_desc_retrieval(
        self,
        task: Task,
        text: str,
        examples: Optional[List[dict]] = None,
        debug_mode: bool = False,
        **kwargs
    ):
        gen_prompt = get_generation_prompt(
            task=task,
            text=text,
            examples=examples,
            idx=0
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        if response is None:
            raise ValueError("Response is None.")
        buggy_code = response
        gen_prompt = get_generation_prompt(
            task=task,
            text=buggy_code,
            examples=examples,
            idx=1
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        query = clean_content(response)
        pos = text
        
        if debug_mode:
            result = {
                "generation_prompt": gen_prompt,
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        else:
            result = {
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        return result
    
    def _gen_for_two_step_not_use_last(
        self,
        task: Task,
        text: str,
        examples: Optional[List[dict]] = None,
        debug_mode: bool = False,
        reverse_query_pos: bool = False,
        **kwargs
    ):
        gen_prompt = get_generation_prompt(
            task=task,
            text=text,
            idx=0
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        query = clean_content(response)
        gen_prompt = get_generation_prompt(
            task=task,
            text=query,
            examples=examples,
            idx=1
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        pos = clean_content(response)
        if reverse_query_pos:
            query, pos = pos, query

        if debug_mode:
            result = {
                "generation_prompt": gen_prompt,
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        else:
            result = {
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        return result

    def _gen_for_two_step_use_last(
        self,
        task: Task,
        text: str,
        examples: Optional[List[dict]] = None,
        debug_mode: bool = False,
        reverse_query_pos: bool = False,
        **kwargs
    ):
        gen_prompt = get_generation_prompt(
            task=task,
            text=text,
            idx=0
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        query = clean_content(response) + f"\n```\n{text}\n```"
        gen_prompt = get_generation_prompt(
            task=task,
            text=query,
            examples=examples,
            idx=1
        )
        response = self.chat(gen_prompt, **kwargs)[0]
        pos = clean_content(response)
        if reverse_query_pos:
            query, pos = pos, query

        if debug_mode:
            result = {
                "generation_prompt": gen_prompt,
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        else:
            result = {
                "prompt": task.task_instruction,
                "query": query,
                "pos": [pos],
                "neg": []
            }
        return result

    def generate_triplets(
        self,
        data: dict,
        task: Task,
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        debug_mode: bool = False,
        **kwargs
    ):
        kwargs["remove_thinking"] = not debug_mode
        
        result_list = []
        
        examples = None
        if examples_pool is not None:
            examples = random.sample(examples_pool, min(num_examples, len(examples_pool)))

        try:
            if task.task_type in SPECIAL_TASK_STEPS:
                text = data["text"]
                
                if task.task_type == TaskType.code_modification_retrieval:
                    text_b = data["similar"][0]
                    
                    result = self._gen_for_code_modification_retrieval(
                        task=task,
                        text=text,
                        text_b=text_b,
                        examples=examples,
                        debug_mode=debug_mode
                    )
                elif task.task_type == TaskType.code_comparison_retrieval:
                    text_b = data["similar"][0]
                    
                    result = self._gen_for_code_comparison_retrieval(
                        task=task,
                        text=text,
                        text_b=text_b,
                        examples=examples,
                        debug_mode=debug_mode
                    )
                elif task.task_type == TaskType.bug_desc_retrieval:
                    result = self._gen_for_bug_desc_retrieval(
                        task=task,
                        text=text,
                        examples=examples,
                        debug_mode=debug_mode
                    )
                elif task.task_type in [
                    # cf - updated
                    TaskType.code_issue_discussion_retrieval,
                    TaskType.code_version_update_retrieval,
                    TaskType.code_bug_fix_example_retrieval,
                ]:
                    result = self._gen_for_two_step_not_use_last(
                        task=task,
                        text=text,
                        examples=examples,
                        debug_mode=debug_mode,
                        reverse_query_pos=False
                    )
                elif task.task_type in [
                    # cf - updated
                    TaskType.code_refactoring_pattern_retrieval,
                    TaskType.code_style_guideline_example_retrieval,
                    TaskType.code_migration_retrieval,
                    # jl - updated
                    TaskType.code_optimization_hybrid_retrieval,
                    TaskType.code_best_practices_retrieval,
                    TaskType.security_vulnerability_fix_retrieval,
                ]:
                    result = self._gen_for_two_step_use_last(
                        task=task,
                        text=text,
                        examples=examples,
                        debug_mode=debug_mode,
                        reverse_query_pos=False
                    )
                else:
                    raise NotImplementedError(f"Task type {task.task_type} not implemented.")
            elif task.task_type == TaskType.code_context_retrieval:
                text = data["text"]
                
                result = self._gen_for_code_context_retrieval(
                    task=task,
                    text=text,
                    **kwargs
                )
                # NOTE: no need to do quality control for code context retrieval task
                result_list.append(result)
                return result_list
            else:
                text = data["text"]
                
                result = self._gen_for_normal_task(
                    task=task,
                    text=text,
                    examples=examples,
                    debug_mode=debug_mode,
                    **kwargs
                )
            
            # print(gen_prompt)
            # print('================================================')
            qc_prompt = get_quality_control_prompt(
                task=task,
                query=result["query"],
                pos=result["pos"][0]
            )
            # print(qc_prompt)
            # print('*********************************************************************')
            response = self.chat(qc_prompt, **kwargs)[0]
            judge = clean_content(response)
            # print(response, judge)
            if "1" in judge:
                if debug_mode:
                    result["judge"] = judge
                    result["judge_response"] = response
                result_list.append(result)
            else:
                if debug_mode:
                    result["judge"] = judge
                    result["judge_response"] = response
                    result_list.append(result)
        except Exception as e:
            warn(f"Error: {e}")
        
        return result_list

    def gen_hard_negatives(self, result: dict, task: Task, num_negatives: int = 7, **kwargs):
        gen_hard_neg_prompt = get_gen_hard_neg_prompt(
            task=task,
            query=result["query"],
            pos=result["pos"][0]
        )
        response_list = self.chat(gen_hard_neg_prompt, n=num_negatives, **kwargs)
        for response in response_list:
            if response is None:
                continue
            hard_neg = clean_content(response)
            result["neg"].append(hard_neg)
        result["neg"] = list(set(result["neg"]))
        return result

    def run_single(
        self,
        data: dict,
        task: Task,
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        debug_mode: bool = False,
        gen_hard_neg: bool = False,
        num_negatives: int = 7,
        **kwargs
    ):
        result_list = []

        docid = compute_md5(data["text"])
        if self.cache_dir is not None:
            gen_data_cache_path = os.path.join(self.cache_dir, f"{docid}.json")
            if os.path.exists(gen_data_cache_path):
                with open(gen_data_cache_path, "r", encoding="utf-8") as f:
                    result_list = json.load(f)
                
                if len(result_list) > 0:
                    if gen_hard_neg:
                        for i in range(len(result_list)):
                            if len(result_list[i]["neg"]) == 0:
                                result_list[i] = self.gen_hard_negatives(
                                    result=result_list[i],
                                    task=task,
                                    num_negatives=num_negatives,
                                    **kwargs
                                )
                        # overwrite the cache file
                        with open(gen_data_cache_path, "w", encoding="utf-8") as f:
                            json.dump(result_list, f, indent=4, ensure_ascii=False)
                    return result_list

        triplets = self.generate_triplets(
            data,
            task=task,
            examples_pool=examples_pool,
            num_examples=num_examples,
            debug_mode=debug_mode,
            **kwargs
        )
        if len(triplets) == 0:
            return []
        
        result = triplets[0]
        if debug_mode:
            result["docid"] = docid
        
        if gen_hard_neg:
            result = self.gen_hard_negatives(
                result,
                task=task,
                num_negatives=num_negatives,
                **kwargs
            )
        
        result_list.append(result)
        
        if self.cache_dir is not None:
            gen_data_cache_path = os.path.join(self.cache_dir, f"{docid}.json")
            with open(gen_data_cache_path, "w", encoding="utf-8") as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)
        
        return result_list

    def run(
        self,
        positives: List[dict],
        task_type: str,
        language: str = "en",
        code_language: str = "python",
        tgt_code_language: Optional[str] = None,
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        tqdm_desc: str = "Generating triplets",
        debug_mode: bool = False,
        gen_hard_neg: bool = False,
        num_negatives: int = 7,
        thread_count: int = 1,
        **kwargs
    ):
        task = get_task(
            task_type=task_type,
            language=language,
            code_language=code_language,
            tgt_code_language=tgt_code_language
        )
        
        result_list = []

        def process_positive(positive):
            return self.run_single(
                data=positive,
                task=task,
                examples_pool=examples_pool,
                num_examples=num_examples,
                debug_mode=debug_mode,
                gen_hard_neg=gen_hard_neg,
                num_negatives=num_negatives,
                **kwargs
            )
        # Use thread pool for parallel processing with tqdm progress bar.
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(tqdm(executor.map(
                process_positive,
                positives
            ), total=len(positives), desc=tqdm_desc))

        # Collect results into result_list.
        for res in results:
            if isinstance(res, list):
                result_list.extend(res)
            else:
                result_list.append(res)
        # result_list.extend(results)

        return result_list

    def run_for_gen_neg(
        self,
        pairs: List[dict],
        task_type: str,
        language: str = "en",
        code_language: str = "python",
        tgt_code_language: Optional[str] = None,
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        tqdm_desc: str = "Generating triplets",
        debug_mode: bool = False,
        gen_hard_neg: bool = False,
        num_negatives: int = 7,
        thread_count: int = 1,
        **kwargs
    ):
        task = get_task(
            task_type=task_type,
            language=language,
            code_language=code_language,
            tgt_code_language=tgt_code_language
        )
        
        result_list = []

        def gen_single_negative(pair):
            result = self.gen_hard_negatives(
                pair,
                task=task,
                num_negatives=num_negatives,
                **kwargs
            )
            return [result]

        # Use thread pool for parallel processing with tqdm progress bar.
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(tqdm(executor.map(
                gen_single_negative,
                pairs
            ), total=len(pairs), desc=tqdm_desc))

        # Collect results into result_list.
        for res in results:
            if isinstance(res, list):
                result_list.extend(res)
            else:
                result_list.append(res)
        # result_list.extend(results)

        return result_list
