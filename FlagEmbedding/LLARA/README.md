<div align="center">
<h1> Llama2Vec: Unsupervised Adaptation of Large Language Models for Dense Retrieval (LLARA) [<a href="https://arxiv.org/abs/2312.15503">paper</a>]</h1>
</div>

Llama2Vec consists of two pretext tasks:
- **EBAE** (Embedding-Based Auto-Encoding)
- **EBAR** (Embedding-Based Auto-Regression)

The LLM is prompted to **reconstruct the input sentence** and **predict the next sentence** based on its text embeddings.

It is known for the following features:
- simple
- lightweight
- highly effective

## Environment
```bash
conda create llara python=3.10

conda activate llara

# You may need to adjust the cuda version
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.41.0 deepspeed accelerate datasets peft pandas
pip install flash-attn --no-build-isolation
```

## Model List

| Model                                                        | Introduction                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BAAI/LLARA-pretrain](https://huggingface.co/BAAI/LLARA-pretrain) | LLARA that has undergone unsupervised adaptation on Wikipedia |
| [BAAI/LLARA-passage](https://huggingface.co/BAAI/LLARA-passage) | The LLARA-pretrain model fine-tuned on MS MARCO passage (the hard negatives come from dense retriever) |
| [BAAI/LLARA-document](https://huggingface.co/BAAI/LLARA-document) | The LLARA-pretrain model fine-tuned on MS MARCO document     |
| [BAAI/LLARA-beir](https://huggingface.co/BAAI/LLARA-beir)    | The LLARA-pretrain model fine-tuned on MS MARCO passage (the hard negatives come from BM25) |

## Usage

```python
import torch
from transformers import AutoModel, AutoTokenizer, LlamaModel

def get_query_inputs(queries, tokenizer, max_length=512):
    prefix = '"'
    suffix = '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>'
    prefix_ids = tokenizer(prefix, return_tensors=None)['input_ids']
    suffix_ids = tokenizer(suffix, return_tensors=None)['input_ids'][1:]
    queries_inputs = []
    for query in queries:
        inputs = tokenizer(query,
                           return_tensors=None,
                           max_length=max_length,
                           truncation=True,
                           add_special_tokens=False)
        inputs['input_ids'] = prefix_ids + inputs['input_ids'] + suffix_ids
        inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        queries_inputs.append(inputs)
    return tokenizer.pad(
            queries_inputs,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

def get_passage_inputs(passages, tokenizer, max_length=512):
    prefix = '"'
    suffix = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
    prefix_ids = tokenizer(prefix, return_tensors=None)['input_ids']
    suffix_ids = tokenizer(suffix, return_tensors=None)['input_ids'][1:]
    passages_inputs = []
    for passage in passages:
        inputs = tokenizer(passage,
                           return_tensors=None,
                           max_length=max_length,
                           truncation=True,
                           add_special_tokens=False)
        inputs['input_ids'] = prefix_ids + inputs['input_ids'] + suffix_ids
        inputs['attention_mask'] = [1] * len(inputs['input_ids'])
        passages_inputs.append(inputs)
    return tokenizer.pad(
            passages_inputs,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('BAAI/LLARA-passage')
model = AutoModel.from_pretrained('BAAI/LLARA-passage')

# Define query and passage inputs
query = "What is llama?"
title = "Llama"
passage = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
query_input = get_query_inputs([query], tokenizer)
passage_input = get_passage_inputs([passage], tokenizer)


with torch.no_grad():
    # compute query embedding
    query_outputs = model(**query_input, return_dict=True, output_hidden_states=True)
    query_embedding = query_outputs.hidden_states[-1][:, -8:, :]
    query_embedding = torch.mean(query_embedding, dim=1)
    query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)

    # compute passage embedding
    passage_outputs = model(**passage_input, return_dict=True, output_hidden_states=True)
    passage_embeddings = passage_outputs.hidden_states[-1][:, -8:, :]
    passage_embeddings = torch.mean(passage_embeddings, dim=1)
    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, dim=-1)

    # compute similarity score
    score = query_embedding @ passage_embeddings.T
    print(score)

```

## Unsupervised Adaption (pretrain)
1. You can get the complete data here: [cfli/pretrain_wiki](https://huggingface.co/datasets/cfli/pretrain_wiki)
2. Here is an example for pretrain:
```shell
cd ./pretrain
torchrun --nproc_per_node 8 \
run.py \
--output_dir ./output \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--train_data ../data/pretrain/toy_pretrain_data.jsonl \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--dataloader_drop_last True \
--cutoff_len 128 \
--logging_steps 1 \
--save_steps 500 \
--save_total_limit 20 \
--gradient_checkpointing \
--ddp_find_unused_parameters False \
--use_flash_attn False \
--deepspeed ../stage1.json \
--warmup_ratio 0.1 \
--remove_stop_words True \
--use_lora False \
--bf16 \
--cache_dir ./LMs \
--token ...
```
If you want to pretrain based on the complete data, please use hype-parameters in our paper.

## Fine-tune

Here is an example for fine-tune:
```shell
cd ./finetune
torchrun --nproc_per_node 8 \
run.py \
--output_dir ./output \
--model_name_or_path BAAI/LLARA-pretrain \
--train_data ../data/finetune/toy_finetune_data.jsonl \
--learning_rate 3e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.01 \
--query_max_len 64 \
--passage_max_len 160 \
--train_group_size 16 \
--logging_steps 10 \
--save_steps 500 \
--save_total_limit 3 \
--ddp_find_unused_parameters False \
--negatives_cross_device \
--gradient_checkpointing \
--deepspeed ../stage1.json \
--warmup_ratio 0.1 \
--fp16 \
--cache_dir ./LMs \
--token ...
```

## Citation

If you find this repository useful, please give us a star ⭐.

To cite our work:

```
@misc{li2023makinglargelanguagemodels,
      title={Making Large Language Models A Better Foundation For Dense Retrieval}, 
      author={Chaofan Li and Zheng Liu and Shitao Xiao and Yingxia Shao},
      year={2023},
      eprint={2312.15503},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.15503}, 
}
```
