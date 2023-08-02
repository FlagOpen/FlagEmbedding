import argparse

from mteb import MTEB
from models import UniversalModel
from C_MTEB import *
from C_MTEB import ChineseTaskList



query_instruction_for_retrieval_dict = {
    "BAAI/baai-general-embedding-large-zh-instruction": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/baai-general-embedding-large-zh": None
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/baai-general-embedding-large-zh-instruction", type=str)
    parser.add_argument('--task_type', default=None, type=str)
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()

    task_names = [t.description["name"] for t in MTEB(task_types=None if args.task_type is None else args.task_type,
                                                      task_langs=['zh']).tasks]

    for task in task_names:
        if task not in ChineseTaskList:
            continue
        if task in ['T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval',
                    'CovidRetrieval', 'CmedqaRetrieval',
                    'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval',
                    'T2Reranking', 'MmarcoReranking', 'CMedQAv1', 'CMedQAv2']:
            instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
        else:
            instruction = None

        model = UniversalModel(model_name_or_path=args.model_name_or_path,
                               normlized=True,
                               query_instruction_for_retrieval=instruction)

        evaluation = MTEB(tasks=[task], task_langs=['zh'])
        evaluation.run(model, output_folder=f"zh_results/{args.model_name_or_path.split('/')[-1]}")



