import logging
import torch

import datasets
from dataclasses import asdict
from transformers import (
    HfArgumentParser,
)
from src.retrieval import DenseRetriever
from src.retrieval.metrics import RetrievalMetric
from src.retrieval.trainer import RetrievalTrainer, EarlyExitCallBack
from src.retrieval.args import RetrievalArgs, RetrievalTrainingArgs
from src.retrieval.data import RetrievalDataset, RetrievalDataCollator, SameDatasetTrainDataset, TASK_CONFIG
from src.utils.util import FileLogger, makedirs

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((RetrievalArgs, RetrievalTrainingArgs))
    model_args, training_args = parser.parse_args_into_dataclasses()
    model_args: RetrievalArgs
    training_args: RetrievalTrainingArgs

    config = TASK_CONFIG[model_args.version]
    instruction = config["instruction"]

    model = DenseRetriever(
        **asdict(model_args), 
        cache_dir=model_args.model_cache_dir, 
        cos_temperature=training_args.cos_temperature,
        contrastive_weight=training_args.contrastive_weight,
        distill_weight=training_args.distill_weight,
        teacher_temperature=training_args.teacher_temperature,
        student_temperature=training_args.student_temperature,
        negative_cross_device=training_args.negative_cross_device,
        stable_distill=training_args.stable_distill,
    )
    # if model_args.train_data is not None:
    #     model.to(torch.float32)
    
    if training_args.use_train_config:
        model.train_config = config["training"]

    tokenizer = model.tokenizer

    with training_args.main_process_first():
        train_dataset, task_indices_range = RetrievalDataset.prepare_train_dataset(
            data_file=model_args.train_data, 
            cache_dir=model_args.dataset_cache_dir,
            add_instruction=model_args.add_instruction,
            train_group_size=training_args.train_group_size,
            config=config,
            use_train_config=training_args.use_train_config,
            select_positive=training_args.select_positive,
            select_negative=training_args.select_negative,
            max_sample_num=training_args.max_sample_num,
            teacher_scores_margin=training_args.teacher_scores_margin,
            teacher_scores_min=training_args.teacher_scores_min,
            stable_distill=training_args.stable_distill,
        )

        # we should get the evaluation task before specifying instruction
        if model_args.eval_data is not None and model_args.add_instruction:
            raw_eval_dataset = datasets.load_dataset('json', data_files=model_args.eval_data, split='train', cache_dir=model_args.dataset_cache_dir)
            eval_task = raw_eval_dataset[0]["task"]
        else:
            eval_task = None

        eval_dataset = RetrievalDataset.prepare_eval_dataset(
            data_file=model_args.eval_data, 
            cache_dir=model_args.dataset_cache_dir,
            instruction=instruction[eval_task] if eval_task is not None else None,
            eval_method=training_args.eval_method,
        )
        corpus = RetrievalDataset.prepare_corpus(
            data_file=model_args.corpus,
            key_template=model_args.key_template,
            cache_dir=model_args.dataset_cache_dir,
            instruction=instruction[eval_task] if eval_task is not None else None 
        )
    
    if training_args.process_index == 0:
        # NOTE: this corpus is for computing metrics, where no instruction is given
        no_instruction_corpus = RetrievalDataset.prepare_corpus(
            data_file=model_args.corpus,
            key_template=model_args.key_template,
            cache_dir=model_args.dataset_cache_dir,
        )
    else:
        no_instruction_corpus = None

    if training_args.inbatch_same_dataset is not None:
        assert training_args.dataloader_num_workers == 0, f"Make sure dataloader num_workers is 0 when using inbatch_same_dataset!"
        train_dataset = SameDatasetTrainDataset(
            train_dataset, 
            task_indices_range, 
            batch_size=training_args.per_device_train_batch_size, 
            seed=training_args.seed, 
            organize_method=training_args.inbatch_same_dataset, 
            num_processes=training_args.world_size,
            process_index=training_args.process_index,
        )
        training_args.per_device_train_batch_size = 1
    
    if training_args.early_exit_steps is not None:
        callbacks = [EarlyExitCallBack(training_args.early_exit_steps)]
    else:
        callbacks = []

    trainer = RetrievalTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        corpus=corpus,
        model_args=model_args,
        data_collator=RetrievalDataCollator(
            tokenizer=tokenizer,
            query_max_length=model_args.query_max_length,
            key_max_length=model_args.key_max_length,
            inbatch_same_dataset=training_args.inbatch_same_dataset
        ),
        compute_metrics=RetrievalMetric.get_metric_fn(
            model_args.metrics,
            # for collecting labels
            eval_data=model_args.eval_data,
            cutoffs=model_args.cutoffs,
            # for collecting positives and collating retrieval results
            save_name=model_args.save_name,
            output_dir=training_args.output_dir,
            save_to_output=model_args.save_to_output,
            # for restoring text from indices when collating results
            corpus=no_instruction_corpus,
            max_neg_num=model_args.max_neg_num,
            # for nq metrics
            cache_dir=model_args.dataset_cache_dir,
            # for collate_neg
            filter_answers=model_args.filter_answers
        ),
        file_logger=FileLogger(makedirs(training_args.log_path))
    )
    # tie accelerators
    model.accelerator = trainer.accelerator

    # Training
    if train_dataset is not None:
        trainer.train()
        return

    if eval_dataset is not None:
        trainer.evaluate()

if __name__ == "__main__":
    main()
