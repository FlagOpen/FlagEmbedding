import argparse
from mteb import MTEB

from C_MTEB.tasks import *
from C_MTEB import ChineseTaskList
from flag_dres_model import FlagDRESModel


query_instruction_for_retrieval_dict = {
    "BAAI/bge-large-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-large-zh-noinstruct": None,
    "BAAI/bge-base-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-small-zh": "为这个句子生成表示以用于检索相关文章：",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-large-zh", type=str)
    parser.add_argument('--task_type', default=None, type=str)
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()

    model = FlagDRESModel(model_name_or_path=args.model_name_or_path,
                          query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")

    task_names = [t.description["name"] for t in MTEB(task_types=args.task_type,
                                                      task_langs=['zh', 'zh-CN']).tasks]

    for task in task_names:
        # if task not in ChineseTaskList:
        #     continue
        if task in ['T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval',
                    'CovidRetrieval', 'CmedqaRetrieval',
                    'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval',
                    'T2Reranking', 'MmarcoReranking', 'CMedQAv1', 'CMedQAv2']:
            if args.model_name_or_path not in query_instruction_for_retrieval_dict:
                # instruction = "为这个句子生成表示以用于检索相关文章："
                instruction = None
                print(f"{args.model_name_or_path} not in query_instruction_for_retrieval_dict, set instruction=None")
            else:
                instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
        else:
            instruction = None

        model.query_instruction_for_retrieval = instruction

        evaluation = MTEB(tasks=[task], task_langs=['zh'])
        evaluation.run(model, output_folder=f"zh_results/{args.model_name_or_path.split('/')[-1]}")



