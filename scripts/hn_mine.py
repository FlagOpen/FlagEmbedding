import json
import random
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field

import faiss
from transformers import HfArgumentParser
from FlagEmbedding import FlagAutoModel
from FlagEmbedding.abc.inference import AbsEmbedder


@dataclass
class DataArgs:
    """
    Data arguments for hard negative mining.
    """
    input_file: str = field(
        metadata={"help": "The input file for hard negative mining."}
    )
    output_file: str = field(
        metadata={"help": "The output file for hard negative mining."}
    )
    candidate_pool: Optional[str] = field(
        default=None, metadata={"help": "The candidate pool for hard negative mining. If provided, it should be a jsonl file, each line is a dict with a key 'text'."}
    )
    range_for_sampling: str = field(
        default="10-210", metadata={"help": "The range to sample negatives."}
    )
    negative_number: int = field(
        default=15, metadata={"help": "The number of negatives."}
    )
    use_gpu_for_searching: bool = field(
        default=False, metadata={"help": "Whether to use faiss-gpu for searching."}
    )
    search_batch_size: int = field(
        default=64, metadata={"help": "The batch size for searching."}
    )


@dataclass
class ModelArgs:
    """
    Model arguments for embedder.
    """
    embedder_name_or_path: str = field(
        metadata={"help": "The embedder name or path.", "required": True}
    )
    embedder_model_class: Optional[str] = field(
        default=None, metadata={"help": "The embedder model class. Available classes: ['encoder-only-base', 'encoder-only-m3', 'decoder-only-base', 'decoder-only-icl']. Default: None. For the custom model, you need to specifiy the model class.", "choices": ["encoder-only-base", "encoder-only-m3", "decoder-only-base", "decoder-only-icl"]}
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "whether to normalize the embeddings"}
    )
    pooling_method: str = field(
        default="cls", metadata={"help": "The pooling method fot the embedder."}
    )
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    devices: Optional[str] = field(
        default=None, metadata={"help": "Devices to use for inference.", "nargs": "+"}
    )
    query_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_retrieval: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    examples_for_task: Optional[str] = field(
        default=None, metadata={"help": "Examples for task"}
    )
    examples_instruction_format: str = field(
        default="{}{}", metadata={"help": "Format for examples instruction"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for models."}
    )
    # ================ for inference ===============
    batch_size: int = field(
        default=3000, metadata={"help": "Batch size for inference."}
    )
    embedder_query_max_length: int = field(
        default=512, metadata={"help": "Max length for query."}
    )
    embedder_passage_max_length: int = field(
        default=512, metadata={"help": "Max length for passage."}
    )


def create_index(embeddings: np.ndarray, use_gpu: bool = False):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(
    index: faiss.Index,
    query: np.ndarray,
    topk: int = 200,
    batch_size: int = 64
):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def get_corpus(candidate_pool: str):
    corpus = []
    with open(candidate_pool, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            corpus.append(line['text'])
    return corpus


def find_knn_neg(
    model: AbsEmbedder,
    input_file: str,
    output_file: str,
    candidate_pool: Optional[str] = None,
    sample_range: str = "10-210",
    negative_number: int = 15,
    use_gpu: bool = False
):
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file):
        line = json.loads(line.strip())
        train_data.append(line)
        corpus.extend(line['pos'])
        if 'neg' in line:
            corpus.extend(line['neg'])
        queries.append(line['query'])

    if candidate_pool is not None:
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus))

    print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
    p_vecs = model.encode(corpus)
    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries)

    print('create index and search------------------')
    index = create_index(p_vecs, use_gpu=use_gpu)
    _, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1])
    assert len(all_inxs) == len(train_data)

    for i, data in enumerate(train_data):
        query = data['query']
        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1: break
            if corpus[inx] not in data['pos'] and corpus[inx] != query:
                filtered_inx.append(inx)

        if len(filtered_inx) > negative_number:
            filtered_inx = random.sample(filtered_inx, negative_number)
        data['neg'] = [corpus[inx] for inx in filtered_inx]

    with open(output_file, 'w') as f:
        for data in train_data:
            if len(data['neg']) < negative_number:
                samples = random.sample(corpus, negative_number - len(data['neg']) + len(data['pos']))
                samples = [sent for sent in samples if sent not in data['pos']]
                data['neg'].extend(samples[: negative_number - len(data['neg'])])
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def load_model(model_args: ModelArgs):
    model = FlagAutoModel.from_finetuned(
        model_name_or_path=model_args.embedder_name_or_path,
        model_class=model_args.embedder_model_class,
        normalize_embeddings=model_args.normalize_embeddings,
        pooling_method=model_args.pooling_method,
        use_fp16=model_args.use_fp16,
        query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
        query_instruction_format=model_args.query_instruction_format_for_retrieval,
        devices=model_args.devices,
        examples_for_task=model_args.examples_for_task,
        examples_instruction_format=model_args.examples_instruction_format,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        batch_size=model_args.batch_size,
        query_max_length=model_args.embedder_query_max_length,
        passage_max_length=model_args.embedder_passage_max_length,
    )
    return model


def main(data_args: DataArgs, model_args: ModelArgs):
    model = load_model(model_args)

    find_knn_neg(
        model=model,
        input_file=data_args.input_file,
        output_file=data_args.output_file,
        candidate_pool=data_args.candidate_pool,
        sample_range=[int(x) for x in data_args.range_for_sampling.split('-')],
        negative_number=data_args.negative_number,
        use_gpu=data_args.use_gpu_for_searching
    )


if __name__ == "__main__":
    parser = HfArgumentParser((
        DataArgs,
        ModelArgs
    ))
    data_args, model_args = parser.parse_args_into_dataclasses()
    data_args: DataArgs
    model_args: ModelArgs
    main(data_args, model_args)
