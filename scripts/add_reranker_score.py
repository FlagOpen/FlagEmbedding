import json

from FlagEmbedding import FlagAutoReranker
from FlagEmbedding.abc.evaluation import AbsEvalModelArgs
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ScoreArgs:
    input_file: str = field(
        default=None, metadata={"help": "The input json file, each line includes query, pos and neg."}
    )
    output_file: str = field(
        default=None, metadata={"help": "The output json file, it includes query, pos, neg, pos_scores and neg_scores."}
    )


if __name__ == '__main__':
    parser = HfArgumentParser((
        ScoreArgs,
        AbsEvalModelArgs
    ))
    score_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: ScoreArgs
    model_args: AbsEvalModelArgs

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

    with open(score_args.output_dir, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')