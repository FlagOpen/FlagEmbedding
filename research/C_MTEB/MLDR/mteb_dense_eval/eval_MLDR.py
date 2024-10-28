"""
python3 eval_MLDR.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--results_save_path ./results \
--max_query_length 512 \
--max_passage_length 8192 \
--batch_size 256 \
--corpus_batch_size 1 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False \
--overwrite False
"""
import os
from mteb import MTEB
from pprint import pprint
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from flag_dres_model import FlagDRESModel
# from mteb.tasks import MultiLongDocRetrieval
from C_MTEB.tasks.MultiLongDocRetrieval import MultiLongDocRetrieval


@dataclass
class EvalArgs:
    results_save_path: str = field(
        default='./results',
        metadata={'help': 'Path to save results.'}
    )
    languages: str = field(
        default=None,
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh', 
                  "nargs": "+"}
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "whether to overwrite evaluation results"}
    )


@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'encoder name or path.'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean', 'last'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add instruction?'}
    )
    query_instruction_for_retrieval: str = field(
        default=None,
        metadata={'help': 'query instruction for retrieval'}
    )
    passage_instruction_for_retrieval: str = field(
        default=None,
        metadata={'help': 'passage instruction for retrieval'}
    )
    max_query_length: int = field(
        default=512,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=8192,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    corpus_batch_size: int = field(
        default=2,
        metadata={'help': 'Inference batch size for corpus. If 0, then use `batch_size`.'}
    )


def check_languages(languages):
    if languages is None:
        return None
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    
    encoder = model_args.encoder
    
    if encoder[-1] == '/':
        encoder = encoder[:-1]
    
    model = FlagDRESModel(
        model_name_or_path=encoder,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        query_instruction_for_retrieval=model_args.query_instruction_for_retrieval if model_args.add_instruction else None,
        passage_instruction_for_retrieval=model_args.passage_instruction_for_retrieval if model_args.add_instruction else None,
        max_query_length=model_args.max_query_length,
        max_passage_length=model_args.max_passage_length,
        batch_size=model_args.batch_size,
        corpus_batch_size=model_args.corpus_batch_size
    )
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)
    output_folder = os.path.join(eval_args.results_save_path, f'{os.path.basename(encoder)}_max-length-{model_args.max_passage_length}')
    
    print("==================================================")
    print("Start evaluating model:")
    print(model_args.encoder)
    
    evaluation = MTEB(tasks=[
        MultiLongDocRetrieval(langs=languages)
    ])
    results_dict = evaluation.run(model, eval_splits=["test"], output_folder=output_folder, overwrite_results=eval_args.overwrite, corpus_chunk_size=200000)
    
    print(output_folder + ":")
    pprint(results_dict)
    
    print("==================================================")
    print("Finish MultiLongDocRetrieval evaluation for model:")
    print(model_args.encoder)


if __name__ == "__main__":
    main()
