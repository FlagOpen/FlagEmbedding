import argparse

from mteb import MTEB
from flag_dres_model import FlagDRESModel


query_instruction_for_retrieval_dict = {
    "BAAI/bge-large-en": "Represent this sentence for searching relevant passages: ",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-large-en", type=str)
    parser.add_argument('--task_type', default=None, type=str, help="task type. Default is None, which means using all task types")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = FlagDRESModel(model_name_or_path=args.model_name_or_path,
                          normalize_embeddings=False,  # normlize embedding will harm the performance of classification task
                          query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")

    task_names = [t.description["name"] for t in MTEB(task_types=None if args.task_type is None else args.task_type,
                                                      task_langs=['en']).tasks]

    for task in task_names:
        if task in ['MSMARCOv2']:
            print('Skip task: {}, since it has no test split'.format(task))
            continue

        if 'CQADupstack' in task or task in ['Touche2020', 'SciFact', 'TRECCOVID', 'NQ',
                                             'NFCorpus', 'MSMARCO', 'HotpotQA', 'FiQA2018',
                                             'FEVER', 'DBPedia', 'ClimateFEVER', 'SCIDOCS', ]:
            if args.model_name_or_path not in query_instruction_for_retrieval_dict:
                # instruction = "Represent this sentence for searching relevant passages: "
                instruction = None
                print(f"{args.model_name_or_path} not in query_instruction_for_retrieval_dict, set instruction=None")
            else:
                instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
        else:
            instruction = None

        model.query_instruction_for_retrieval = instruction

        evaluation = MTEB(tasks=[task], task_langs=['en'])
        evaluation.run(model, output_folder=f"en_results/{args.model_name_or_path.split('/')[-1]}")



