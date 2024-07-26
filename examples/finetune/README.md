# Finetune
In this example, we show how to finetune the baai-general-embedding with your data.

## 1. Installation
* **with pip**
```
pip install -U FlagEmbedding
```

* **from source**
```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install  .
```
For development, install as editable:
```
pip install -e .
```

 

## 2. Data format
Train data should be a json file, where each line is a dict like this:

```
{"query": str, "pos": List[str], "neg":List[str]}
```

`query` is the query, and `pos` is a list of positive texts, `neg` is a list of negative texts.
If you have no negative texts for a query, you can random sample some from the entire corpus as the negatives.

See [toy_finetune_data.jsonl](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/toy_finetune_data.jsonl) for a toy data file.

### Hard Negatives 

Hard negatives is a widely used method to improve the quality of sentence embedding. 
You can mine hard negatives following this command:
```bash
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```

- `input_file`: json data for finetuning. This script will retrieve top-k documents for each query, 
and random sample negatives from the top-k documents (not including the positive documents).
- `output_file`: path to save JSON data with mined hard negatives for finetuning
- `negative_number`: the number of sampled negatives 
- `range_for_sampling`: where to sample negative. For example, `2-100` means sampling `negative_number` negatives from top2-top200 documents. **You can set larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top60-300 passages)**
- `candidate_pool`: The pool to retrieval. The default value is None, and this script will retrieve from the combination of all `neg` in `input_file`. 
The format of this file is the same as [pretrain data](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain#2-data-format). If input a candidate_pool, this script will retrieve negatives from this file.
- `use_gpu_for_searching`: whether to use faiss-gpu to retrieve negatives.


## 3. Train
```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-zh-v1.5 \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {large batch size; set 1 for toy data} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "" 
```

**some important arguments**:
- `per_device_train_batch_size`: batch size in training. In most of cases, larger batch size will bring stronger performance. You can expand it by enabling `--fp16`, `--deepspeed ./df_config.json` (df_config.json can refer to [ds_config.json](./ds_config.json)), `--gradient_checkpointing`, etc. 
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one positive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.
- `learning_rate`: select a appropriate for your model. Recommend 1e-5/2e-5/3e-5 for large/base/small-scale. 
- `temperature`: It will influence the distribution of similarity scores. **Recommended value: 0.01-0.1.**
- `query_max_len`: max length for query. Please set it according the average length of queries in your data.
- `passage_max_len`: max length for passage. Please set it according the average length of passages in your data.
- `query_instruction_for_retrieval`: instruction for query, which will be added to each query. You also can set it `""` to add nothing to query.
- `use_inbatch_neg`: use passages in the same batch as negatives. Default value is True. 
- `save_steps`: for setting how many training steps to save a checkpoint.

For more training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)


### 4. Model merging via [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail) [optional]

For more details please refer to [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail).

Fine-tuning the base bge model can improve its performance on target task, 
but maybe lead to severe degeneration of model’s general capabilities 
beyond the targeted domain (e.g., lower performance on c-mteb tasks). 
By merging the fine-tuned model and the base model, 
LM-Cocktail can significantly enhance performance in downstream task
while maintaining performance in other unrelated tasks.

```python
from LM_Cocktail import mix_models, mix_models_with_data

# Mix fine-tuned model and base model; then save it to output_path: ./mixed_model_1
model = mix_models(
    model_names_or_paths=["BAAI/bge-large-en-v1.5", "your_fine-tuned_model"], 
    model_type='encoder', 
    weights=[0.5, 0.5],  # you can change the weights to get a better trade-off.
    output_path='./mixed_model_1')
```

If you have a new task, and there is no data or resource can be used for fine-tuning, 
you can try to use LM-Cocktail to merge existing models (from open-source community or your models fine-tuned on other tasks) to produce a task-specific model. 
In this way, you just need to construct a few example data and don't need fine-tuning the base model.
For example, you can merge the models from [huggingface](https://huggingface.co/Shitao) using the example data for your task:
```python
from LM_Cocktail import mix_models, mix_models_with_data

example_data = [
    {"query": "How does one become an actor in the Telugu Film Industry?", "pos": [" How do I become an actor in Telugu film industry?"], "neg": [" What is the story of Moses and Ramesses?", " Does caste system affect economic growth of India?"]}, 
    {"query": "Why do some computer programmers develop amazing software or new concepts, while some are stuck with basic programming work?", "pos": [" Why do some computer programmers develops amazing softwares or new concepts, while some are stuck with basics programming works?"], "neg": [" When visiting a friend, do you ever think about what would happen if you did something wildly inappropriate like punch them or destroy their furniture?", " What is the difference between a compliment and flirting?"]}
]

model = mix_models_with_data(
    model_names_or_paths=["BAAI/bge-base-en-v1.5", "Shitao/bge-hotpotqa", "Shitao/bge-quora"], 
    model_type='encoder', 
    example_ata=example_data,
    temperature=5.0,
    max_input_length=512,
    neg_number=2)
```
**Since there are only 9 `bge-*` models in this [repo](https://huggingface.co/Shitao), the performance may not be satisfactory when your task is different with all 9 fine-tuning tasks. 
You can fine-tune the base model on more tasks and merge them to achieve better performance on your task.**


### 5. Load your model
After fine-tuning BGE model, you can load it easily in the same way as [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding#usage) 

Please replace the `query_instruction_for_retrieval` with your instruction if you set a different value for hyper-parameter `--query_instruction_for_retrieval` when fine-tuning.


### 6. Evaluate model
We provide [a simple script](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding/finetune/eval_msmarco.py) to evaluate the model's performance.
A brief summary of how the script works:
1. Load the model on all available GPUs through [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html). 
2. Encode the corpus and offload the embeddings in `faiss` Flat index. By default, `faiss` also dumps the index on all available GPUs.
3. Encode the queries and search `100` nearest neighbors for each query.
4. Compute Recall and MRR metrics.

First, install `faiss`, a popular approximate nearest neighbor search library:
```bash
conda install -c conda-forge faiss-gpu
```

#### 6.1 MSMARCO dataset
The default evaluate data is MSMARCO, a widely used retrieval benchmark.

You can check the data formats for the [msmarco corpus](https://huggingface.co/datasets/namespace-Pt/msmarco-corpus) and [evaluation queries](https://huggingface.co/datasets/namespace-Pt/msmarco). 

Run the following command:

```bash
python -m FlagEmbedding.baai_general_embedding.finetune.eval_msmarco \
--encoder BAAI/bge-base-en-v1.5 \
--fp16 \
--add_instruction \
--k 100
```
**some important arguments:**
- `encoder`: specify the encoder model, which can be either a model on huggingface or a local one.
- `fp16`: use half precision for inference.
- `add_instruction`: add retrieval instruction (`Represent this sentence for searching relevant passages: `).
- `k`: specify how many nearest neighbors to retrieve for each query.

The results should be similar to
```python
{
    'MRR@1': 0.2330945558739255, 
    'MRR@10': 0.35786976395142633, 
    'MRR@100': 0.3692618036917553, 
    'Recall@1': 0.22606255969436478, 
    'Recall@10': 0.6412965616045848, 
    'Recall@100': 0.9012774594078318
}
```

#### 6.2 Your dataset

You should prepare two files with jsonl format: 
- One is corpus_data, which contains the text you want to search. A toy example: [toy_corpus.json](./toy_evaluation_data/toy_corpus.json)
```
{"content": "A is ..."}
{"content": "B is ..."}
{"content": "C is ..."}
{"content": "Panda is ..."}
{"content": "... is A"}
```
- The other is query_data, which contains the queries and the ground truth. A toy example: [toy_corpus.json](./toy_evaluation_data/toy_query.json)
```
{"query": "What is A?", "positive": ["A is ...", "... is A"]}
{"query": "What is B?", "positive": ["B is ..."]}
{"query": "What is C?", "positive": ["C is ..."]}
```

Then, pass the data path to evaluation script: 
```bash
python -m FlagEmbedding.baai_general_embedding.finetune.eval_msmarco \
--encoder BAAI/bge-base-en-v1.5 \
--fp16 \
--add_instruction \
--k 100 \
--corpus_data ./toy_evaluation_data/toy_corpus.json \
--query_data ./toy_evaluation_data/toy_query.json 
```


