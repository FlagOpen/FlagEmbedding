import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import ModelArguments, DataArguments
from data import TrainDatasetForCE, GroupCollator, TrainDatasetForCL
from modeling import CLEncoder, CLProjEncoder, CrossEncoder
from trainer import CETrainer

logger = logging.getLogger(__name__)
from pprint import pprint as pp
import sys 
sys.path.append("/opt/tiger/FlagEmbedding")
from utils import get_complete_last_checkpoint
import transformers
import os
os.environ["WANDB_DISABLED"]="true"

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

# 记录similarity pos and neg
# class Loggingback(TrainerCallback):
#     def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
#         if state.global_step % 10 == 0:
#             control.

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # for args in (model_args, data_args, training_args): pp(args)

    # check and load checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_complete_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.info(
                f"Output directory ({training_args.output_dir}) already exists and is empty."
                "Train from scratch"
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
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

    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True
    )

    if model_args.model_type=="CrossEncoder": _model_class = CrossEncoder
    elif model_args.model_type == "CLEncoder": _model_class = CLEncoder
    else: _model_class = CLProjEncoder
        
    if model_args.model_type=="CrossEncoder": train_dataset = TrainDatasetForCE(data_args, tokenizer=tokenizer)
    else: train_dataset = TrainDatasetForCL(data_args, tokenizer=tokenizer)

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        logger.info(f"train start from {training_args.resume_from_checkpoint}")
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        logger.info(f"train start from {last_checkpoint}")
        checkpoint = last_checkpoint

    _trainer_class = CETrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
        tokenizer=tokenizer
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # trainer.train()
    # trainer.save_model()


if __name__ == "__main__":
    main()
