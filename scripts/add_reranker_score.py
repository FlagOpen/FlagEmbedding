import json
from typing import Optional, List

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from FlagEmbedding import FlagAutoReranker


@dataclass
class ScoreArgs:
    input_file: str = field(
        default=None, metadata={"help": "The input jsonl file, each line includes query, pos and neg."}
    )
    output_file: str = field(
        default=None, metadata={"help": "The output jsonl file, it includes query, pos, neg, pos_scores and neg_scores."}
    )


@dataclass
class ModelArgs:
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    devices: Optional[str] = field(
        default=None, metadata={"help": "Devices to use for inference.", "nargs": "+"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code"}
    )
    reranker_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The reranker name or path."}
    )
    reranker_model_class: Optional[str] = field(
        default=None, metadata={"help": "The reranker model class. Available classes: ['encoder-only-base', 'decoder-only-base', 'decoder-only-layerwise', 'decoder-only-lightweight']. Default: None. For the custom model, you need to specify the model class.", "choices": ["encoder-only-base", "decoder-only-base", "decoder-only-layerwise", "decoder-only-lightweight"]}
    )
    reranker_peft_path: Optional[str] = field(
        default=None, metadata={"help": "The reranker peft path."}
    )
    use_bf16: bool = field(
        default=False, metadata={"help": "whether to use bf16 for inference"}
    )
    query_instruction_for_rerank: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_rerank: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    passage_instruction_for_rerank: Optional[str] = field(
        default=None, metadata={"help": "Instruction for passage"}
    )
    passage_instruction_format_for_rerank: str = field(
        default="{}{}", metadata={"help": "Format for passage instruction"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for models."}
    )
    # ================ for inference ===============
    reranker_batch_size: int = field(
        default=3000, metadata={"help": "Batch size for inference."}
    )
    reranker_query_max_length: Optional[int] = field(
        default=None, metadata={"help": "Max length for reranking."}
    )
    reranker_max_length: int = field(
        default=512, metadata={"help": "Max length for reranking."}
    )
    normalize: bool = field(
        default=False, metadata={"help": "whether to normalize the reranking scores"}
    )
    prompt: Optional[str] = field(
        default=None, metadata={"help": "The prompt for the reranker."}
    )
    cutoff_layers: List[int] = field(
        default=None, metadata={"help": "The output layers of layerwise/lightweight reranker."}
    )
    compress_ratio: int = field(
        default=1, metadata={"help": "The compress ratio of lightweight reranker."}
    )
    compress_layers: Optional[int] = field(
        default=None, metadata={"help": "The compress layers of lightweight reranker.", "nargs": "+"}
    )


def main(score_args: ScoreArgs, model_args: ModelArgs):
    reranker = FlagAutoReranker.from_finetuned(
        model_name_or_path=model_args.reranker_name_or_path,
        model_class=model_args.reranker_model_class,
        peft_path=model_args.reranker_peft_path,
        use_fp16=model_args.use_fp16,
        use_bf16=model_args.use_bf16,
        query_instruction_for_rerank=model_args.query_instruction_for_rerank,
        query_instruction_format=model_args.query_instruction_format_for_rerank,
        passage_instruction_for_rerank=model_args.passage_instruction_for_rerank,
        passage_instruction_format=model_args.passage_instruction_format_for_rerank,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        devices=model_args.devices,
        normalize=model_args.normalize,
        prompt=model_args.prompt,
        cutoff_layers=model_args.cutoff_layers,
        compress_layers=model_args.compress_layers,
        compress_ratio=model_args.compress_ratio,
        batch_size=model_args.reranker_batch_size,
        query_max_length=model_args.reranker_query_max_length,
        max_length=model_args.reranker_max_length,
    )

    pairs = []
    data = []
    with open(score_args.input_file) as f:
        for line in f:
            data.append(json.loads(line))
            for p in data[-1]['pos']:
                pairs.append((data[-1]['query'], p))
            for p in data[-1]['neg']:
                pairs.append((data[-1]['query'], p))

    scores = reranker.compute_score(pairs)

    score_idx = 0
    for i in range(len(data)):
        data[i]['pos_scores'] = []
        data[i]['neg_scores'] = []
        for _ in range(len(data[i]['pos'])):
            data[i]['pos_scores'].append(float(scores[score_idx]))
            score_idx += 1
        for _ in range(len(data[i]['neg'])):
            data[i]['neg_scores'].append(float(scores[score_idx]))
            score_idx += 1

    with open(score_args.output_file, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')


if __name__ == "__main__":
    parser = HfArgumentParser((
        ScoreArgs,
        ModelArgs
    ))
    score_args, model_args = parser.parse_args_into_dataclasses()
    score_args: ScoreArgs
    model_args: ModelArgs
    main(score_args, model_args)
