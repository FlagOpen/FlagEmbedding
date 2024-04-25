import socket
import contextlib
import os
import argparse
import sys
import torch

def find_unused_port(start_port=37624):
    """
    寻找一个从start_port开始的未使用的端口。
    """
    for port in range(start_port, start_port + 20000):  # 检查1000个端口
        with contextlib.suppress(Exception):
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

# 构建训练命令
command = f"""
torchrun --master-port={port} --nproc_per_node {num_gpus} /opt/tiger/FlagEmbedding/FlagEmbedding/reranker/run.py {args}
"""

# 执行命令
os.system(command)