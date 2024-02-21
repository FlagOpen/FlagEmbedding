# Finetune cross-encoder
In this example, we show how to finetune the cross-encoder reranker with your data.

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

The data format for reranker is the same as [embedding fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#data-format).
Besides, we strongly suggest to [mine hard negatives](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives) to fine-tune reranker.


## 3. Train

```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.reranker.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-reranker-base \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {batch size; set 1 for toy data} \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--train_group_size 16 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 10 
```

**some important arguments**:
- `per_device_train_batch_size`: batch size in training. 
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one positive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.

More training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)


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
    model_names_or_paths=["BAAI/bge-reranker-base", "your_fine-tuned_model"], 
    model_type='reranker', 
    weights=[0.5, 0.5],  # you can change the weights to get a better trade-off.
    output_path='./mixed_model_1')
```




### 5. Load your model

#### Using FlagEmbedding

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True) #use fp16 can speed up computing

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```


#### Using Huggingface transformers

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```





