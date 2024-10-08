import socket
import contextlib
import os
import sys
import torch
os.environ["WANDB_DISABLED"]="true"
args = (" ").join(sys.argv)
# 使用示例
num_gpus = torch.cuda.device_count()
os.system("cd /opt/tiger/FlagEmbedding")
if not os.path.exists("/opt/tiger/train_15neg"): os.system("cp -r /mnt/bn/data-tns-live-llm/leon/experiments/llm/fcbank/train_15neg /opt/tiger/train_15neg")

#——————————————————————————————————————————————————debug——————————————————————————————————————————————————————————#
# args = "--output_dir /mnt/bn/data-tns-live-llm/leon/experiments/llm/fcbank/toy --model_name_or_path /mnt/bn/data-tns-live-llm/leon/experiments/llm/fcbank/xlmr/models--FacebookAI--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/ --train_data /opt/tiger/FlagEmbedding/examples/finetune/toy_finetune_data.jsonl --learning_rate 1e-5 --fp16 --num_train_epochs 5 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --dataloader_drop_last --train_group_size 10 --max_len 512 --weight_decay 0.01 --logging_steps 10 --save_strategy epoch --save_steps 1 --save_total_limit 3"
# print(args)
# num_gpus = 1
#——————————————————————————————————————————————————debug——————————————————————————————————————————————————————————#

# 构建训练命令
command = f"""torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node {num_gpus} {args}"""
# 执行命令
os.system(command)