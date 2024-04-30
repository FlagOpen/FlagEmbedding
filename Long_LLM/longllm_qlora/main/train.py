import torch
import logging
from transformers import HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from src import ( 
    Data,
    DefaultDataCollator,
    ModelArgs,
    FileLogger,
    get_model_and_tokenizer,
    makedirs,
    format_numel_str
)
from src.args import TrainingArgs
from src.metrics import Metric
from src.trainer import LLMTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = get_model_and_tokenizer(model_args, return_tokenizer_only=True, evaluation_mode=False)

    # NOTE: must import here, otherwise raise invalid device error
    from unsloth import FastLanguageModel
    if model_args.load_in_4_bit:
        device_map = None
    else:
        device_map = {"": "cuda"}
    
    model, _ = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = model_args.max_length,
        dtype = torch.bfloat16,
        device_map=device_map,
        load_in_4bit = model_args.load_in_4_bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        token=model_args.access_token,
        cache_dir=model_args.model_cache_dir,

        rope_theta=model_args.rope_theta,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = training_args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = training_args.lora_targets,
        modules_to_save=training_args.lora_extra_params,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    print(model.config)

    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

    with training_args.main_process_first():
        train_dataset = Data.prepare_train_data(
            model_args.train_data, 
            tokenizer=tokenizer,
            max_length=model_args.max_length,
            min_length=training_args.min_length,
            chat_template=model_args.chat_template,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )

    with training_args.main_process_first():
        if is_deepspeed_zero3_enabled() and training_args.eval_method != "perplexity":
            logger.warning(f"In deepspeed zero3, evaluation with generation is may lead to hang because of the unequal number of forward passes across different devices.")
        eval_dataset = Data.prepare_eval_data(
            model_args.eval_data, 
            tokenizer=tokenizer,
            max_length=training_args.eval_max_length,
            min_length=training_args.eval_min_length,
            chat_template=model_args.chat_template,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )

    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(tokenizer),
        file_logger=FileLogger(makedirs(training_args.log_path)),
        compute_metrics=Metric.get_metric_fn(
            metrics=training_args.metrics,
            save_path=Metric.get_save_path(
                model_args.eval_data,
                training_args.output_dir
            ) if model_args.eval_data is not None else None
        )
    )
    if train_dataset is not None:
        trainer.train()
    elif eval_dataset is not None:
        trainer.evaluate()

if __name__ == "__main__":
    main()
