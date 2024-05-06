import logging
from transformers import (
    HfArgumentParser, 
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from src import ( 
    Data,
    DefaultDataCollator,
    ModelArgs,
    TrainingArgs,
    FileLogger,
    get_model_and_tokenizer,
    makedirs
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()
    
    model, tokenizer = get_model_and_tokenizer(model_args)

    for name, param in model.named_parameters():
        if "beacon" not in name:
            param.requires_grad_(False)

    if training_args.lora_tune:
        # copied from LongLoRA
        config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_targets,
            modules_to_save=training_args.lora_extra_params,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    with training_args.main_process_first():
        train_dataset = Data.prepare_train_data(
            model_args.train_data, 
            tokenizer=tokenizer,
            max_length=model_args.max_length,
            min_length=training_args.min_length,
            max_train_num_per_data=training_args.max_train_num_per_data,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )
        eval_dataset = Data.prepare_eval_data(
            model_args.eval_data, 
            tokenizer=tokenizer,
            max_length=training_args.eval_max_length,
            min_length=training_args.eval_min_length,
            max_eval_num=training_args.max_eval_num,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )

    if training_args.use_colossal:
        from src.colossal import ColossalFoldLlamaTrainer as Trainer
    else:
        from src.trainer import FoldLlamaTrainer as Trainer

    # if training_args.curriculum_steps is not None:
    #     from src.trainer import CurriculumFoldCallBack
    #     callbacks = [CurriculumFoldCallBack()]
    # else:
    #     callbacks = None

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        # callbacks=callbacks,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(tokenizer),
        file_logger=FileLogger(makedirs(training_args.log_path))
    )
    if train_dataset is not None:
        trainer.train()
    elif eval_dataset is not None:
        trainer.evaluate()

if __name__ == "__main__":
    main()
