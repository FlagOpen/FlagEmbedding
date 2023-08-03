import argparse

from mteb import MTEB
from models import UniversalModel


query_instruction_for_retrieval_dict = {
    "BAAI/baai-general-embedding-large-en-instruction": "Represent this sentence for searching relevant passages: ",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/baai-general-embedding-large-en-instruction", type=str)
    parser.add_argument('--task_type', default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    task_names = [t.description["name"] for t in MTEB(task_types=None if args.task_type is None else args.task_type,
                                                      task_langs=['en']).tasks]

    for task in task_names:
        if task in ['MSMARCOv2']:
            print('Skip task: {}, since it has no test split'.format(task))
            continue
        if 'CQADupstack' in task or task in ['Touche2020', 'SciFact', 'TRECCOVID', 'NQ',
                                             'NFCorpus', 'MSMARCO', 'HotpotQA', 'FiQA2018',
                                             'FEVER', 'DBPedia', 'ClimateFEVER', 'SCIDOCS', ]:
            instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
        else:
            instruction = None

        model = UniversalModel(model_name_or_path=args.model_name_or_path,
                               normlized=False,
                               query_instruction_for_retrieval=instruction)

        evaluation = MTEB(tasks=[task], task_langs=['zh'])
        evaluation.run(model, output_folder=f"en_results/{args.model_name_or_path.split('/')[-1]}")



