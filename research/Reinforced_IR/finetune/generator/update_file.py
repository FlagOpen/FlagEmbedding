import argparse
import os
import json
import copy


def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--info_path', type=str, default=None)

    opt = parser.parse_args()

    return opt


def main(opt):
    input_path = opt.input_path
    output_path = opt.output_path
    file_name = opt.file_name
    info_path = opt.info_path

    if os.path.exists(info_path):
        info = json.load(open(info_path, 'r'))
    else:
        info = {}

    os.makedirs('/'.join(info_path.split('/')[:-1]), exist_ok=True)

    info[file_name] = {
        "file_name": "train_llamafactory.jsonl",
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "chosen_score": "chosen_score",
            "reject_score": "reject_score"
        }
    }

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    data_format = {
        "conversations": [
            {
                "from": "human",
                "value": ""
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": ""
        },
        "rejected": {
            "from": "gpt",
            "value": ""
        },
        "chosen_score": 0,
        "rejected_score": 0
    }

    data = []
    with open(input_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            tmp_data = copy.deepcopy(data_format)
            tmp_data['conversations'][0]['value'] = line['prompt'].replace('\n###Response:\n', '').replace(
                '###Instruction:\n', '')
            tmp_data['chosen']['value'] = line['chosen']
            tmp_data['rejected']['value'] = line['rejected']
            tmp_data['chosen_score'] = line['chosen_score']
            tmp_data['rejected_score'] = line['rejected_score']
            data.append(tmp_data)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    opt = parse_option()
    main(opt)