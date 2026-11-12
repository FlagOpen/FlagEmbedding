import jsonlines
import csv
import os

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def save_tsv_dict(data, fp, fields):
    # build dir
    dir_path = os.path.dirname(fp)
    os.makedirs(dir_path, exist_ok=True)
    
    # writing to csv file
    with open(fp, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter='\t',)
        writer.writeheader()
        writer.writerows(data)

def cost_esitmate(path):
    corpus = load_jsonlines(os.path.join(path, "corpus.jsonl"))
    queries = load_jsonlines(os.path.join(path, "queries.jsonl"))
    num_corpus_words = 0
    num_queries_words = 0
    for item in tqdm(corpus):
        num_corpus_words += len(item["text"].split(" "))
    for item in tqdm(queries):
        num_queries_words += len(item["text"].split(" "))
    print(len(corpus))
    print(len(queries))
    print(num_corpus_words)
    print(num_queries_words)
