import json
import os
import subprocess

import numpy as np
import torch
from arguments import ModelArguments, DataArguments
from datasets import load_dataset
from torch.utils.data import Dataset, SequentialSampler
from torch_geometric.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, is_torch_npu_available
from transformers import PreTrainedTokenizer, AutoModel


class EmbDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            path: str
    ):
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentences = self.data[item]['contents']
        batch_dict = self.tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
        attention_mask = batch_dict['attention_mask'][0].tolist() + [0] * (512 - len(batch_dict['attention_mask'][0]))
        token_type_ids = batch_dict['token_type_ids'][0].tolist() + [0] * (512 - len(batch_dict['token_type_ids'][0]))
        input_ids = batch_dict['input_ids'][0].tolist() + [0] * (512 - len(batch_dict['token_type_ids'][0]))

        return torch.LongTensor(input_ids), torch.LongTensor(token_type_ids), torch.LongTensor(attention_mask)


def inference(json_path, emb_path, model_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif is_torch_npu_available():
        device = torch.device("npu")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModel.from_pretrained(model_path).to(device)
    model = torch.nn.parallel.DataParallel(model)

    dataset = EmbDataset(tokenizer, json_path)
    loader = DataLoader(dataset=dataset, batch_size=2048, sampler=SequentialSampler(dataset), shuffle=False,
                        drop_last=False, num_workers=16)

    model.eval()
    existing_data = []
    for step, data in enumerate(tqdm(loader, total=len(loader))):
        input_ids, token_type_ids, attention_mask = data
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            batch_vecs = outputs[0][:, 0]
            batch_vecs = torch.nn.functional.normalize(batch_vecs, p=2, dim=-1).detach().cpu().numpy()

            existing_data.append(batch_vecs)

    np.save(emb_path, np.concatenate(existing_data))


def build_bm25_index(dataset, collection_path, index_path):
    title = dataset['title']
    text = dataset['text']
    json_list = []
    for i in range(len(title)):
        json_dict = {'id': i, 'contents': title[i] + ' -- ' + text[i]}
        json_list.append(json_dict)

    with open(os.path.join(collection_path, 'documents.json'), 'w') as f:
        json.dump(json_list, f)

    command = f"python -u -m pyserini.index.lucene   --collection JsonCollection   --input {collection_path}  --index {index_path}  --generator DefaultLuceneDocumentGenerator   --threads 8   --storePositions --storeDocvectors --storeRaw"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        output = result.stdout
        print("execute successful!")
        print(output)
    else:
        print("execute false!")
        print(result.stderr)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    dataset_path = os.path.join(data_args.data_path, 'dataset')
    collection_path = os.path.join(data_args.data_path, 'collection')
    index_path = os.path.join(data_args.data_path, 'index')
    emb_path = os.path.join(data_args.data_path, 'emb')
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(collection_path, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)
    os.makedirs(emb_path, exist_ok=True)
    dataset = load_dataset(f"Cohere/wikipedia-22-12", 'zh', split='train')
    dataset.save_to_disk(dataset_path)
    build_bm25_index(dataset, collection_path, index_path)
    inference(os.path.join(collection_path, 'documents.json'),
              os.path.join(emb_path, 'data.npy'),
              model_args.model_name_or_path)
