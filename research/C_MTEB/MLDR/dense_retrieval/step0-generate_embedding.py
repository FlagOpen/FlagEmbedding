"""
python step0-generate_embedding.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--index_save_dir ./corpus-index \
--max_passage_length 8192 \
--batch_size 4 \
--fp16 \
--pooling_method cls \
--normalize_embeddings True
"""
import os
import faiss
import datasets
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagModel
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 in inference?'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh', 
                  "nargs": "+"}
    )
    index_save_dir: str = field(
        default='./corpus-index',
        metadata={'help': 'Dir to save index. Corpus index will be saved to `index_save_dir/{encoder_name}/{lang}/index`. Corpus ids will be saved to `index_save_dir/{encoder_name}/{lang}/docid` .'}
    )
    max_passage_length: int = field(
        default=512,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )


def get_model(model_args: ModelArgs):
    model = FlagModel(
        model_args.encoder, 
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.fp16
    )
    return model


def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages


def load_corpus(lang: str):    
    corpus = datasets.load_dataset('Shitao/MLDR', f'corpus-{lang}', split='corpus')
    
    corpus_list = [{'id': e['docid'], 'content': e['text']} for e in tqdm(corpus, desc="Generating corpus")]
    corpus = datasets.Dataset.from_list(corpus_list)
    return corpus


def generate_index(model: FlagModel, corpus: datasets.Dataset, max_passage_length: int=512, batch_size: int=256):
    corpus_embeddings = model.encode_corpus(corpus["content"], batch_size=batch_size, max_length=max_passage_length)
    dim = corpus_embeddings.shape[-1]
    
    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index, list(corpus["id"])


def save_result(index: faiss.Index, docid: list, index_save_dir: str):
    docid_save_path = os.path.join(index_save_dir, 'docid')
    index_save_path = os.path.join(index_save_dir, 'index')
    with open(docid_save_path, 'w', encoding='utf-8') as f:
        for _id in docid:
            f.write(str(_id) + '\n')
    faiss.write_index(index, index_save_path)


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    
    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]
    
    model = get_model(model_args=model_args)
    
    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)
    
    print("==================================================")
    print("Start generating embedding with model:")
    print(model_args.encoder)

    print('Generate embedding of following languages: ', languages)
    for lang in languages:
        print("**************************************************")
        index_save_dir = os.path.join(eval_args.index_save_dir, os.path.basename(encoder), lang)
        if not os.path.exists(index_save_dir):
            os.makedirs(index_save_dir)
        if os.path.exists(os.path.join(index_save_dir, 'index')) and not eval_args.overwrite:
            print(f'Embedding of {lang} already exists. Skip...')
            continue
        
        print(f"Start generating embedding of {lang} ...")
        corpus = load_corpus(lang)
        
        index, docid = generate_index(
            model=model, 
            corpus=corpus,
            max_passage_length=eval_args.max_passage_length,
            batch_size=eval_args.batch_size
        )
        save_result(index, docid, index_save_dir)

    print("==================================================")
    print("Finish generating embeddings with model:")
    print(model_args.encoder)


if __name__ == "__main__":
    main()
