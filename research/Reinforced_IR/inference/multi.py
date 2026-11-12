import os
import json
import shutil
import multiprocessing
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class Args():
    generate_model_path: str = field(
        default='Meta-Llama-3-8B',
        metadata={"help": "Generate model path"}
    )
    generate_model_lora_path: str = field(
        default=None,
        metadata={"help": "Generate model path"}
    )
    temperature: float = field(
        default=0.8,
        metadata={"help": "Temperature for generation"}
    )
    gpu_memory_utilization: float = field(
        default=0.8,
        metadata={"help": "GPU memory utilization"}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top p for generation"}
    )
    max_tokens: int = field(
        default=300,
        metadata={"help": "Max tokens for generation"}
    )
    model_type: str = field(
        default='llm_instruct',
        metadata={"help": "LLM model type"}
    )
    input_dir: str = field(
        default=None,
        metadata={"help": "Input directory", "required": True}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Output directory", "required": True}
    )
    num_gpus: int = field(
        default=8,
        metadata={"help": "Number of GPUs"}
    )
    rm_tmp: bool = field(
        default=True,
        metadata={"help": "Remove temporary files"}
    )
    start_gpu: int = field(
        default=0
    )


def worker_function(device):
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{device}'

    import torch
    from agent import LLMInstructAgent, LLMAgent

    print("Available GPUs:", torch.cuda.device_count())
    print("Device:", device)

    num_splits = args.num_gpus

    input_dir = args.input_dir
    output_split_dir = os.path.join(args.output_dir, f'tmp_split_{device}')
    
    os.makedirs(output_split_dir, exist_ok=True)
    
    if args.model_type == 'llm':
        llm = LLMAgent(generate_model_path=args.generate_model_path,
                       gpu_memory_utilization=args.gpu_memory_utilization)
    else:
        llm = LLMInstructAgent(generate_model_path=args.generate_model_path,
                               gpu_memory_utilization=args.gpu_memory_utilization)
    
    print("========================================")
    for file in os.listdir(input_dir):
        if not file.endswith('.json'):
            continue
        
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_split_dir, file)
        
        prompt_list = json.load(open(input_path, 'r', encoding='utf-8'))

        length = len(prompt_list)
        start = int(device) * length // num_splits
        if int(device) == num_splits - 1:
            end = length
        else:
            end = (int(device) + 1) * length // num_splits

        split_prompt_list = prompt_list[start : end]
        
        print('----------------------------------------')
        print(f"Processing {input_path} on device {device}, {args.model_type}")
        
        output_list = llm.generate(split_prompt_list,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   max_tokens=args.max_tokens,
                                   stop=[],
                                #    lora_path=args.generate_model_lora_path
                                   )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, indent=4)
        
        print(f"Finished {input_path} on device {device}")


def merge(args: Args):
    num_splits = args.num_gpus

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_split_dir_list = [os.path.join(output_dir, f'tmp_split_{i}') for i in range(num_splits)]
    
    for file in os.listdir(input_dir):
        if not file.endswith('.json'):
            continue
        
        merged_output_list = []
        for output_split_dir in output_split_dir_list:
            # output_split_dir = output_split_dir_list[i]
            output_path = os.path.join(output_split_dir, file)
            merged_output_list.extend(json.load(open(output_path, 'r', encoding='utf-8')))

        merged_output_path = os.path.join(output_dir, file)
        if os.path.exists(merged_output_path):
            merged_output_list.extend(json.load(open(merged_output_path)))
        with open(merged_output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_output_list, f, indent=4)
    
    if args.rm_tmp:
        for output_split_dir in output_split_dir_list:
            shutil.rmtree(output_split_dir)
            print(f"Removed {output_split_dir}")
    
    print("Finished merging")


if __name__ == "__main__":
    processes = []
    multiprocessing.set_start_method('spawn')
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    for i in range(args.num_gpus):
        process = multiprocessing.Process(target=worker_function, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    merge(args=args)