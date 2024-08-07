import logging
import os
import sys
from pathlib import Path

import torch
import transformers

from transformers import AutoTokenizer, HfArgumentParser, set_seed, AutoConfig, Trainer

from arguments import ModelArguments, DataArguments, \
    PretrainTrainingArguments as TrainingArguments
from data import TrainDatasetForEmbedding, EmbedCollator
from load_model import get_model
from modeling import PreModel
from trainer import PreTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    model = get_model(model_args, training_args.gradient_checkpointing)

    tokenizer = AutoTokenizer.from_pretrained(
        # model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # special_tokens_dict = {'additional_special_tokens': ['<s1>', '<s2>', '<s3>', '<s4>',
    #                                                      '<s5>', '<s6>', '<s7>', '<s8>',
    #                                                      '<s9>', '<s10>', '<s11>', '<s12>',
    #                                                      '<s13>', '<s14>', '<s15>', '<s16>', ]}
    # tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.padding_side = "left"  # Allow batched inference
    print(tokenizer)

    special_tokens = ['<s1>', '<s2>', '<s3>', '<s4>',
                      '<s5>', '<s6>', '<s7>', '<s8>',
                      '<s9>', '<s10>', '<s11>', '<s12>',
                      '<s13>', '<s14>', '<s15>', '<s16>', ]
    current_vocab = tokenizer.get_vocab()
    tokens_to_add = [token for token in special_tokens if token not in current_vocab]
    if tokens_to_add:
        special_tokens_dict = {'additional_special_tokens': tokens_to_add}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    print(tokenizer)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataset = TrainDatasetForEmbedding(tokenizer, args=data_args)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=EmbedCollator(
            tokenizer=tokenizer,
            cutoff_len=data_args.cutoff_len,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    if training_args.deepspeed:
        trainer.deepspeed.save_checkpoint(training_args.output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    main()