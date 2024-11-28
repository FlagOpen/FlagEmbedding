from transformers import HfArgumentParser
from dataclasses import dataclass
import sys
import os
from typing import Optional
import transformers

# 定义多个参数类
@dataclass
class ModelArguments:
    model_name_or_path: str = "bert-base-uncased"
    cache_dir: Optional[str] = None

@dataclass
class DataArguments:
    sample_num: int = 1000

@dataclass
class TrainingArguments:
    learning_rate: float = 1e-5
    
def main():
    # 创建解析器时传入多个类
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # 支持从 JSON 文件加载配置
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 从 JSON 文件解析
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 从命令行解析
        args = parser.parse_args_into_dataclasses()

    # 解析参数返回元组
    model_args, data_args, training_args = args
    # 或者
    # args = parser.parse_args_into_dataclasses()
    # model_args = args[0]  # ModelArgs 实例
    # training_args = args[1]  # TrainingArgs 实例
    
    # 为什么使用元组返回
    # 不可变性：元组是不可变的，确保参数解析后不会被意外修改
    # 多返回值：Python 使用元组作为多返回值的标准方式
    # 解构支持：Python 支持元组解构赋值，使代码更简洁：

    print(model_args)
    print(data_args)
    print(training_args)

if __name__ == "__main__":
    main()