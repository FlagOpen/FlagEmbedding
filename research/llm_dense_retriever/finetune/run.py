import logging
import os
import torch
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from data import SameDatasetTrainDataset, SameEmbedCollator
from modeling import BiEncoderModel
from trainer import BiTrainer
from load_model import get_model, save_merged_model

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        add_eos_token=True
    )

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    # else:
    #     tokenizer.padding_side = 'right'
    if data_args.use_special_tokens:
        special_tokens_dict = {'additional_special_tokens': ['<instruct>', '<query>', '<response>']}
        add_num = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        add_num = 0
    if add_num > 0:
        resize = True
    else:
        resize = False
    base_model = get_model(model_args, training_args.output_dir, resize, len(tokenizer))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    logger.info('Config: %s', config)

    model = BiEncoderModel(model=base_model,
                           tokenizer=tokenizer,
                           normlized=training_args.normlized,
                           negatives_cross_device=training_args.negatives_cross_device,
                           temperature=training_args.temperature,
                           sub_batch_size=training_args.sub_batch_size)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # if data_args.use_same_batch:
    train_dataset = SameDatasetTrainDataset(args=data_args,
                                            batch_size=training_args.per_device_train_batch_size,
                                            seed=training_args.seed,
                                            tokenizer=tokenizer,
                                            num_processes=training_args.world_size,
                                            process_index=training_args.process_index)
    training_args.per_device_train_batch_size = 1
    training_args.dataloader_num_workers = 0

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SameEmbedCollator(
            tokenizer=tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            sub_batch_size=training_args.sub_batch_size
        ),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        # os.makedirs(os.path.join(training_args.output_dir, 'embedding'), exist_ok=True)
        # torch.save(base_model.model.model.embed_tokens, os.path.join(training_args.output_dir, 'embedding', 'emb.pth'))

def save_model():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if model_args.save_merged_lora_model and training_args.process_index == 0:
        save_merged_model(model_args, training_args.output_dir)

if __name__ == "__main__":
    main()
    save_model()