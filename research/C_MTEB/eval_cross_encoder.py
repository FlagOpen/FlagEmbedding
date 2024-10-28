import argparse

from C_MTEB.tasks import *
from mteb import MTEB

from FlagEmbedding import FlagReranker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-reranker-base", type=str)
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()

    model = FlagReranker(args.model_name_or_path, use_fp16=True)

    if 'checkpoint-' in args.model_name_or_path:
        save_name = "_".join(args.model_name_or_path.split('/')[-2:])
    else:
        save_name = "_".join(args.model_name_or_path.split('/')[-1:])

    evaluation = MTEB(task_types=["Reranking"], task_langs=['zh', 'zh2en', 'en2zh'])
    evaluation.run(model, output_folder=f"reranker_results/{save_name}")



