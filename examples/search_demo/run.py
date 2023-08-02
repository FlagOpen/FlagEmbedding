# encoding=gbk
import os
import sys

sys.path.append('../')

from transformers import HfArgumentParser

from search_demo.tool import LocalDatasetLoader, BMVectorIndex, Agent
from search_demo.arguments import ModelArguments, DataArguments

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    loader = LocalDatasetLoader(data_path=os.path.join(data_args.data_path, 'dataset'),
                                embedding_path=os.path.join(data_args.data_path, 'emb/data.npy'))
    index = BMVectorIndex(model_path=model_args.model_name_or_path,
                          bm_index_path=os.path.join(data_args.data_path, 'index'),
                          data_loader=loader)
    agent = Agent(index)
    while True:
        question = input("问：").strip()
        if question != '':
            agent.answer(question, RANKING=1000, TOP_N=5, verbose=True)
        else:
            break
