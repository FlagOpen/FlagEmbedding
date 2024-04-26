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

args = (" ").join(sys.argv[1:])
print(args)

# 使用示例
port = find_unused_port()
if port:
    print(f"Found an unused port: {port}")
else:
    print("No unused port found.")

num_gpus = torch.cuda.device_count()

os.system("cd /opt/tiger/FlagEmbedding")

if not os.path.exists("/opt/tiger/train_15neg"): os.system("cp -r /mnt/bn/data-tns-live-llm/leon/experiments/llm/fcbank/train_15neg /opt/tiger/train_15neg")

# 构建训练命令
command = f"""
torchrun --master-port={port} --nproc_per_node {num_gpus} /opt/tiger/FlagEmbedding/FlagEmbedding/reranker/run.py {args}
"""

# 执行命令
os.system(command)