"""
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--encoded_query_and_corpus_save_dir ./encoded_query-and-corpus \
--result_save_dir ./search_results \
--threads 16 \
--hits 1000
"""
import os
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
    )


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh', 
                  "nargs": "+"}
    )
    encoded_query_and_corpus_save_dir: str = field(
        default='./encoded_query-and-corpus',
        metadata={'help': 'Dir to save encoded queries and corpus. Encoded queries and corpus are saved in `save_dir/{encoder_name}/{lang}/query_embd.tsv` and `save_dir/{encoder_name}/{lang}/corpus/corpus_embd.jsonl`, individually.'}
    )
    result_save_dir: str = field(
        default='./search_results',
        metadata={'help': 'Dir to saving results. Search results will be saved to `result_save_dir/{encoder_name}/{lang}.txt`'}
    )
    batch_size: int = field(
        default=32,
        metadata={'help': 'Batch size to use during search'}
    )
    threads: int = field(
        default=1,
        metadata={'help': 'Maximum threads to use during search'}
    )
    hits: int = field(
        default=1000,
        metadata={'help': 'Number of hits'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )


def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages


def generate_index(lang: str, corpus_embd_dir: str, index_save_dir: str, threads: int=12):    
    cmd = f"python -m pyserini.index.lucene \
            --language {lang} \
            --collection JsonVectorCollection \
            --input {corpus_embd_dir} \
            --index {index_save_dir} \
            --generator DefaultLuceneDocumentGenerator \
            --threads {threads} \
            --impact --pretokenized --optimize \
        "
    os.system(cmd)


def search_and_save_results(index_save_dir: str, query_embd_path: str, result_save_path: str, batch_size: int = 32, threads: int = 12, hits: int = 1000):
    cmd = f"python -m pyserini.search.lucene \
            --index {index_save_dir} \
            --topics {query_embd_path} \
            --output {result_save_path} \
            --output-format trec \
            --batch {batch_size} \
            --threads {threads} \
            --hits {hits} \
            --impact \
        "
    os.system(cmd)


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    
    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]
    
    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)
    
    print("==================================================")
    print("Start generating search results with model:")
    print(model_args.encoder)

    print('Generate search results of following languages: ', languages)
    for lang in languages:
        print("**************************************************")
        print(f"Start searching results of {lang} ...")
        
        result_save_path = os.path.join(eval_args.result_save_dir, os.path.basename(encoder), f"{lang}.txt")
        if not os.path.exists(os.path.dirname(result_save_path)):
            os.makedirs(os.path.dirname(result_save_path))
        
        if os.path.exists(result_save_path) and not eval_args.overwrite:
            print(f'Search results of {lang} already exists. Skip...')
            continue
        
        encoded_query_and_corpus_save_dir = os.path.join(eval_args.encoded_query_and_corpus_save_dir, os.path.basename(encoder), lang)
        if not os.path.exists(encoded_query_and_corpus_save_dir):
            raise FileNotFoundError(f"{encoded_query_and_corpus_save_dir} not found")
        
        corpus_embd_dir = os.path.join(encoded_query_and_corpus_save_dir, 'corpus')
        index_save_dir = os.path.join(eval_args.encoded_query_and_corpus_save_dir, os.path.basename(encoder), lang, 'index')
        if os.path.exists(index_save_dir) and not eval_args.overwrite:
            print(f'Index of {lang} already exists')
        else:
            generate_index(
                lang=lang,
                corpus_embd_dir=corpus_embd_dir,
                index_save_dir=index_save_dir,
                threads=eval_args.threads
            )
        
        query_embd_path = os.path.join(encoded_query_and_corpus_save_dir, 'query_embd.tsv')
        
        search_and_save_results(
            index_save_dir=index_save_dir,
            query_embd_path=query_embd_path,
            result_save_path=result_save_path,
            batch_size=eval_args.batch_size,
            threads=eval_args.threads,
            hits=eval_args.hits
        )
    
    print("==================================================")
    print("Finish generating search results with model:")
    print(model_args.encoder)


if __name__ == "__main__":
    main()
