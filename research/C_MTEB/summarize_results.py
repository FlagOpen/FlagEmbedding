import argparse
import json
import os
from collections import defaultdict

from C_MTEB import *
import mteb
from mteb import MTEB


CMTEB_tasks = [
    'TNews', 'IFlyTek', 'MultilingualSentiment', 'JDReview', 'OnlineShopping', 'Waimai',
    'CLSClusteringS2S.v2', 'CLSClusteringP2P.v2', 'ThuNewsClusteringS2S.v2', 'ThuNewsClusteringP2P.v2',
    'Ocnli', 'Cmnli',
    'T2Reranking', 'MMarcoReranking', 'CMedQAv1-reranking', 'CMedQAv2-reranking',
    'T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval', 'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval',
    'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STSB', 'AFQMC', 'QBQTC'
]


def read_results(task_types, args):
    tasks_results = {}
    # model_dirs = {}
    for t_type in task_types:
        tasks_results[t_type] = {}
        for t in mteb.get_tasks(task_types=[t_type]):
            task_name = t.metadata.name
            if task_name not in CMTEB_tasks:
                continue

            metric = t.metadata.main_score
            tasks_results[t_type][task_name] = defaultdict(None)

            if os.path.exists(os.path.join(args.results_dir, task_name + '.json')):
                data = json.load(open(os.path.join(args.results_dir, task_name + '.json')))
                for s in ['test', 'dev', 'validation']:
                    if s in data['scores']:
                        split = s
                        break

                temp_data = data['scores'][split][0]
                tasks_results[t_type][task_name] = round(temp_data[metric] * 100, 2)

    return tasks_results


def output_markdown(tasks_results, model, save_file):
    task_type_res = {}
    with open(save_file, 'w') as f:
        for t_type, type_results in tasks_results.items():
            has_CQADupstack = False
            task_cnt = 0
            task_type_res[t_type] = defaultdict()
            f.write(f'Task Type: {t_type}  \n')
            first_line = "| Model |"
            second_line = "|:-------------------------------|"
            for task_name in type_results.keys():
                if "CQADupstack" in task_name:
                    has_CQADupstack = True
                    continue
                first_line += f" {task_name} |"
                second_line += ":--------:|"
                task_cnt += 1
            if has_CQADupstack:
                first_line += f" CQADupstack |"
                second_line += ":--------:|"
                task_cnt += 1
            f.write(first_line + ' Avg |  \n')
            f.write(second_line + ':--------:|  \n')

            write_line = f"| {model} |"
            all_res = []
            cqa_res = []
            for task_name, results in type_results.items():
                if "CQADupstack" in task_name:
                    if model in results:
                        cqa_res.append(results[model])
                    continue

                write_line += f" {results} |"
                all_res.append(results)

            if len(cqa_res) > 0:
                write_line += f" {round(sum(cqa_res) / len(cqa_res), 2)} |"
                all_res.append(round(sum(cqa_res) / len(cqa_res), 2))

            # if len(all_res) == len(type_results.keys()):
            if len(all_res) == task_cnt:
                write_line += f" {round(sum(all_res) / len(all_res), 2)} |"
                task_type_res[t_type][model] = all_res
            else:
                write_line += f"  |"
            f.write(write_line + '  \n\n')

        f.write(f'Overall  \n')
        first_line = "| Model |"
        second_line = "|:-------------------------------|"
        for t_type in task_type_res.keys():
            first_line += f" {t_type} |"
            second_line += ":--------:|"
        f.write(first_line + ' Avg |  \n')
        f.write(second_line + ':--------:|  \n')

        write_line = f"| {model} |"
        all_res = []
        for type_name, results in task_type_res.items():
            if model in results:
                write_line += f" {round(sum(results[model]) / len(results[model]), 2)} |"
                all_res.extend(results[model])
            else:
                write_line += f"  |"

        if len(all_res) > 0:
            write_line += f" {round(sum(all_res) / len(all_res), 2)} |"

        f.write(write_line + '  \n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default="./zh_results", type=str)
    parser.add_argument('--lang', default="zh", type=str)
    parser.add_argument('--model', default="model", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.lang == 'zho':
        task_types = ["Retrieval", "STS", "PairClassification", "Classification", "Reranking", "Clustering"]
        args.lang = ['zho']
    elif args.lang == 'eng':
        task_types = ["Retrieval", "Clustering", "PairClassification", "Reranking", "STS", "Summarization",
                      "Classification"]
        args.lang = ['eng']
    else:
        raise NotImplementedError(f"args.lang must be zh or en, but{args.lang}")

    task_results = read_results(task_types, args=args)

    output_markdown(task_results, args.model,
                    save_file=os.path.join(args.results_dir, f'{args.lang[0]}_results.md'))
