import socket
import contextlib
import os
import sys
import torch

def find_unused_port(start_port=37625):
    """
    寻找一个从start_port开始的未使用的端口。
    """
    for port in range(start_port, start_port + 20000):  # 检查1000个端口
        with contextlib.suppress(Exception):
            for i in range(10):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('', port))
                return port
    return None  # 如果没有找到，返回None

# args = (" ").join(sys.argv[1:])

script_name = "embedding_run"

# 使用示例
port = find_unused_port()
if port:
    print(f"Found an unused port: {port}")
else:
    print("No unused port found.")

num_gpus = torch.cuda.device_count()

os.system("cd /opt/tiger/FlagEmbedding")

if not os.path.exists("/opt/tiger/train_15neg"): os.system("cp -r /mnt/bn/data-tns-live-llm/leon/experiments/llm/fcbank/train_15neg /opt/tiger/train_15neg")

#——————————————————————————————————————————————————debug——————————————————————————————————————————————————————————#
args = "--output_dir /mnt/bn/data-tns-live-llm/leon/experiments/llm/fcbank/toy --model_name_or_path /mnt/bn/data-tns-live-llm/leon/experiments/llm/fcbank/xlmr/models--FacebookAI--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/ --train_data /opt/tiger/FlagEmbedding/examples/finetune/toy_finetune_data.jsonl --learning_rate 1e-5 --fp16 --num_train_epochs 5 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --dataloader_drop_last --train_group_size 10 --max_len 512 --weight_decay 0.01 --logging_steps 10 --save_strategy epoch --save_steps 1 --save_total_limit 3"
print(args)
num_gpus = 1
#——————————————————————————————————————————————————debug——————————————————————————————————————————————————————————#

# 构建训练命令
command = f"""
torchrun --master-port={port} --nproc_per_node {num_gpus} /opt/tiger/FlagEmbedding/FlagEmbedding/reranker/{script_name}.py {args}
"""

# 执行命令
os.system(command)