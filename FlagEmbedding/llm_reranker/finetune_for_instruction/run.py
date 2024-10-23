import logging
import os
from pathlib import Path
import wandb
import random
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from .data import TrainDatasetForReranker, RerankCollator
from .modeling import BiEncoderModel
from .trainer import BiTrainer
from .load_model import get_model

logger = logging.getLogger(__name__)

def try_init_wandb():
    try:
        import wandb
        # 检查环境变量是否有WANDB_API_KEY，这是可选的，根据你的需求决定是否需要
        if os.getenv("WANDB_API_KEY"):
            wandb.init()
            logger.info("W&B initialized.")
        else:
            logger.info("WANDB_API_KEY not found. Skipping W&B initialization.")
    except ImportError:
        logger.info("wandb library not found. Skipping W&B initialization.")
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    try_init_wandb()
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
    base_model = get_model(model_args, training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        trust_remote_code=True,
        token=model_args.token,
        add_eos_token=True
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eod_id is not None:
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token_id = tokenizer.im_end_id
    if 'mistral' in model_args.model_name_or_path.lower():
        tokenizer.padding_side = 'left'

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )
    logger.info('Config: %s', config)

    model = BiEncoderModel(model=base_model,
                           tokenizer=tokenizer,
                           train_batch_size=training_args.per_device_train_batch_size)

    # model = base_model

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataset = TrainDatasetForReranker(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RerankCollator(
            tokenizer=tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
        tokenizer=tokenizer,
    )
    trainer.use_lora = model_args.use_lora

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()

    if not model_args.use_lora:
        checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-final")
        trainer.deepspeed.save_checkpoint(checkpoint_dir)
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
