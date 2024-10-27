import argparse
import json
import os
from collections import defaultdict

from C_MTEB import *
from mteb import MTEB


def read_results(task_types, except_tasks, args):
    tasks_results = {}
    model_dirs = {}
    for t_type in task_types:
        tasks_results[t_type] = {}
        for t in MTEB(task_types=[t_type], task_langs=args.lang).tasks:
            task_name = t.description["name"]
            if task_name in except_tasks: continue

            metric = t.description["main_score"]
            tasks_results[t_type][task_name] = defaultdict(None)

            for model_name in os.listdir(args.results_dir):
                model_dir = os.path.join(args.results_dir, model_name)
                if not os.path.isdir(model_dir): continue
                model_dirs[model_name] = model_dir
                if os.path.exists(os.path.join(model_dir, task_name + '.json')):
                    data = json.load(open(os.path.join(model_dir, task_name + '.json')))
                    for s in ['test', 'dev', 'validation']:
                        if s in data:
                            split = s
                            break

                    if 'en' in args.lang:
                        if 'en-en' in data[split]:
                            temp_data = data[split]['en-en']
                        elif 'en' in data[split]:
                            temp_data = data[split]['en']
                        else:
                            temp_data = data[split]
                    elif 'zh' in args.lang:
                        if 'zh' in data[split]:
                            temp_data = data[split]['zh']
                        elif 'zh-CN' in data[split]:
                            temp_data = data[split]['zh-CN']
                        else:
                            temp_data = data[split]

                    if metric == 'ap':
                        tasks_results[t_type][task_name][model_name] = round(temp_data['cos_sim']['ap'] * 100, 2)
                    elif metric == 'cosine_spearman':
                        tasks_results[t_type][task_name][model_name] = round(temp_data['cos_sim']['spearman'] * 100, 2)
                    else:
                        tasks_results[t_type][task_name][model_name] = round(temp_data[metric] * 100, 2)

    return tasks_results, model_dirs


def output_markdown(tasks_results, model_names, save_file):
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

            for model in model_names:
                write_line = f"| {model} |"
                all_res = []
                cqa_res = []
                for task_name, results in type_results.items():
                    if "CQADupstack" in task_name:
                        if model in results:
                            cqa_res.append(results[model])
                        continue

                    if model in results:
                        write_line += f" {results[model]} |"
                        all_res.append(results[model])
                    else:
                        write_line += f"  |"

                if len(cqa_res) > 0:
                    write_line += f" {round(sum(cqa_res) / len(cqa_res), 2)} |"
                    all_res.append(round(sum(cqa_res) / len(cqa_res), 2))

                # if len(all_res) == len(type_results.keys()):
                if len(all_res) == task_cnt:
                    write_line += f" {round(sum(all_res) / len(all_res), 2)} |"
                    task_type_res[t_type][model] = all_res
                else:
                    write_line += f"  |"
                f.write(write_line + '  \n')

        f.write(f'Overall  \n')
        first_line = "| Model |"
        second_line = "|:-------------------------------|"
        for t_type in task_type_res.keys():
            first_line += f" {t_type} |"
            second_line += ":--------:|"
        f.write(first_line + ' Avg |  \n')
        f.write(second_line + ':--------:|  \n')

        for model in model_names:
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
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.lang == 'zh':
        task_types = ["Retrieval", "STS", "PairClassification", "Classification", "Reranking", "Clustering"]
        except_tasks = []
        args.lang = ['zh', 'zh-CN']
    elif args.lang == 'en':
        task_types = ["Retrieval", "Clustering", "PairClassification", "Reranking", "STS", "Summarization",
                      "Classification"]
        except_tasks = ['MSMARCOv2']
        args.lang = ['en']
    else:
        raise NotImplementedError(f"args.lang must be zh or en, but{args.lang}")

    task_results, model_dirs = read_results(task_types, except_tasks, args=args)

    output_markdown(task_results, model_dirs.keys(),
                    save_file=os.path.join(args.results_dir, f'{args.lang[0]}_results.md'))


