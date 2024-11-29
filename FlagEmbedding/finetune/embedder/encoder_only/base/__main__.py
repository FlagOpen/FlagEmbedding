from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.encoder_only.base import (
    EncoderOnlyEmbedderDataArguments,
    EncoderOnlyEmbedderTrainingArguments,
    EncoderOnlyEmbedderModelArguments,
    EncoderOnlyEmbedderRunner,
)


def main():
    parser = HfArgumentParser((
        EncoderOnlyEmbedderModelArguments,
        EncoderOnlyEmbedderDataArguments,
        EncoderOnlyEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: EncoderOnlyEmbedderModelArguments
    data_args: EncoderOnlyEmbedderDataArguments
    training_args: EncoderOnlyEmbedderTrainingArguments

    runner = EncoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    
    # # DEBUG
    # import os
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["RANK"] = "0" 
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29500"

    # # 然后初始化进程组
    # import torch.distributed as dist
    # if not dist.is_initialized():
    #     dist.init_process_group(
    #         backend='nccl',  # 或者用 'gloo' 
    #         init_method='env://',
    #         world_size=1,
    #         rank=0
    #     )
    
    main()
