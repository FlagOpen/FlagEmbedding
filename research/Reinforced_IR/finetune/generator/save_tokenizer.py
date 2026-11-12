import argparse
import os
import json
import copy

from transformers import AutoTokenizer


def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)

    opt = parser.parse_args()

    return opt


def main(opt):
    model_path = opt.model_path
    output_path = opt.output_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    opt = parse_option()
    main(opt)