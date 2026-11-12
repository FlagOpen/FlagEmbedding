from .generate_prompts import (
    generate_prompt,
    generate_passage_prompt,
    llama_query_generate_prompt,
    llama_answer_generate_prompt,
    llama_generate_train_answer_prompt
)
from .train_prompts import (
    generate_train_answer,
    generate_train_query,
    generate_train_query_type2
)

from .get_prompts import (
    get_query_generation_prompt,
    get_additional_info_generation_prompt,
    get_additional_info_generation_long_prompt,
    get_additional_info_generation_long_air_prompt,
    get_additional_info_generation_train_prompt,
    TASK_DICT,
    get_quality_control_prompt
)

from .teacher_prompts import (
    get_yes_prompt,
    rank_prompt,
    get_rank_prompt
)

__all__ = [
    "generate_prompt",
    "generate_passage_prompt",
    "llama_query_generate_prompt",
    "llama_answer_generate_prompt",
    "llama_generate_train_answer_prompt",
    "generate_train_answer",
    "generate_train_query",
    "generate_train_query_type2",
    "get_query_generation_prompt",
    "get_additional_info_generation_prompt",
    "get_additional_info_generation_long_prompt",
    "get_additional_info_generation_long_air_prompt",
    "get_additional_info_generation_train_prompt",
    "TASK_DICT",
    "get_quality_control_prompt",
    "get_yes_prompt",
    "rank_prompt",
    "get_rank_prompt"
]