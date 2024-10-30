# Reranker

- [Model List](#model-list)
- [Usage](#usage)
- [Citation](#citation)

Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. 
You can get a relevance score by inputting query and passage to the reranker. 
And the score can be mapped to a float value in [0,1] by sigmoid function.

For more detailed using, you can look [reranker-encoder only](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/reranker/encoder_only) or [reranker-decoder only](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/reranker/decoder_only)

## Model List

| Model                                                                     | Base model                                                           | Language | layerwise |                           feature                            |
|:--------------------------------------------------------------------------|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | Chinese and English |     -     | Lightweight reranker model, easy to deploy, with fast inference. |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) | Chinese and English |     -     | Lightweight reranker model, easy to deploy, with fast inference. |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | [bge-m3](https://huggingface.co/BAAI/bge-m3) |    Multilingual     |     -     | Lightweight reranker model, possesses strong multilingual capabilities, easy to deploy, with fast inference. |
| [BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma) | [gemma-2b](https://huggingface.co/google/gemma-2b) |    Multilingual     |     -     | Suitable for multilingual contexts, performs well in both English proficiency and multilingual capabilities. |
| [BAAI/bge-reranker-v2-minicpm-layerwise](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise) | [MiniCPM-2B-dpo-bf16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16) |    Multilingual     |   8-40    | Suitable for multilingual contexts, performs well in both English and Chinese proficiency, allows freedom to select layers for output, facilitating accelerated inference. |


You can select the model according your senario and resource. 
- For **multilingual**, utilize [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) and [BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma)

- For **Chinese or English**, utilize [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) and [BAAI/bge-reranker-v2-minicpm-layerwise](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise). 

- For **efficiency**, utilize [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) and the low layer of [BAAI/bge-reranker-v2-minicpm-layerwise](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise). 

- For better performance, recommand [BAAI/bge-reranker-v2-minicpm-layerwise](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise) and [BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma)

## Usage 
### Using FlagEmbedding

#### 1. Auto Reranker

You can use `FlagAutoReranker` to load the model. For the **custom model** (not included in [`AUTO_RERANKER_MAPPING`](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/reranker/model_mapping.py#L31)), you must specify the `model_class` parameter. You can also submit a pull request to add your **released model** to the [`AUTO_RERANKER_MAPPING`](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/reranker/model_mapping.py#L31) dictionary. If need, you can create a new `<model>.py` file in [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/reranker/encoder_only) or [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/reranker/decoder_only).

```python
from FlagEmbedding import FlagAutoReranker
reranker = FlagAutoReranker.from_finetuned('BAAI/bge-reranker-large',
                                           query_max_length=256,
                                           passage_max_length=512,
                                           use_fp16=True,
                                           devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
score = reranker.compute_score(['query', 'passage'])
print(score) # -1.5263671875

# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
score = reranker.compute_score(['query', 'passage'], normalize=True)
print(score) # 0.1785258315203034

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores) # [-5.60546875, 5.76171875]

# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], normalize=True)
print(scores) # [0.0036642203307843528, 0.9968641641227171]
```

For your **custom model** (assume the model is finetuned from `BAAI/bge-reranker-large`, then the model class is `encoder-only-base`), you can use the following code:

```python
from FlagEmbedding import FlagAutoReranker
reranker = FlagAutoReranker.from_finetuned('your_model_name_or_path',
                                           model_class='encoder-only-base',
                                           query_max_length=256,
                                           passage_max_length=512,
                                           use_fp16=True,
                                           devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
score = reranker.compute_score(['query', 'passage'])
print(score)
```

The `model_class` parameter currently includes the following options:
- `encoder-only-base`: for encoder-only reranker model, such as `BAAI/bge-reranker-large`
- `decoder-only-base`: for decoder-only reranker model, such as `BAAI/bge-reranker-v2-gemma`
- `decoder-only-layerwise`: for decoder-only layerwise reranker model, such as `BAAI/bge-reranker-v2-minicpm-layerwise`
- `decoder-only-lightweight`: for decoder-only lightweight reranker model, such as `BAAI/bge-reranker-v2.5-gemma2-lightweight`

#### 2. Normal Reranker

For `FlagReranker`, it supports `BAAI/bge-reranker-base`, `BAAI/bge-reranker-large`, `BAAI/bge-reranker-v2-m3`:

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker(
    'BAAI/bge-reranker-v2-m3', 
    query_max_length=256,
    passage_max_length=512,
    use_fp16=True,
    devices=['cuda:1']
) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score) # -5.65234375

# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
score = reranker.compute_score(['query', 'passage'], normalize=True)
print(score) # 0.003497010252573502

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores) # [-8.1875, 5.26171875]

# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], normalize=True)
print(scores) # [0.00027803096387751553, 0.9948403768236574]
```

#### 3. LLM-based Reranker

For `FlagLLMReranker`, it supports `BAAI/bge-reranker-v2-gemma`:

```python
from FlagEmbedding import FlagLLMReranker
reranker = FlagLLMReranker(
    'BAAI/bge-reranker-v2-gemma', 
    query_max_length=256,
    passage_max_length=512,
    use_fp16=True,
    devices=['cuda:1']
) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```

#### 4. LLM-based Layerwise Reranker

For `LayerWiseFlagLLMReranker`, it supports `BAAI/bge-reranker-v2-minicpm-layerwise`:

```python
from FlagEmbedding import LayerWiseFlagLLMReranker
reranker = LayerWiseFlagLLMReranker(
    'BAAI/bge-reranker-v2-minicpm-layerwise', 
    query_max_length=256,
    passage_max_length=512,
    use_fp16=True,
    devices=['cuda:1']
) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'], cutoff_layers=[28]) # Adjusting 'cutoff_layers' to pick which layers are used for computing the score.
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], cutoff_layers=[28])
print(scores)
```

#### 5. LLM-based lightweight Reranker

For `LightWeightFlagLLMReranker`, it supports `BAAI/bge-reranker-v2.5-gemma2-lightweight`:

```python
from FlagEmbedding import LightWeightFlagLLMReranker
reranker = LightWeightFlagLLMReranker(
    'BAAI/bge-reranker-v2.5-gemma2-lightweight', 
    query_max_length=256,
    passage_max_length=512,
    use_fp16=True,
    devices=['cuda:1']
) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'], cutoff_layers=[28], compress_ratio=2, compress_layers=[24, 40]) # Adjusting 'cutoff_layers' to pick which layers are used for computing the score.
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], cutoff_layers=[28], compress_ratio=2, compress_layers=[24, 40])
print(scores)
```

### Using Huggingface transformers

#### 1. Normal Reranker

It supports `BAAI/bge-reranker-base`, `BAAI/bge-reranker-large`, `BAAI/bge-reranker-v2-m3`:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```

#### 2. LLM-based reranker

It supports `BAAI/bge-reranker-v2-gemma`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-gemma')
yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = get_inputs(pairs, tokenizer)
    scores = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
    print(scores)
```

#### 3. LLM-based layerwise reranker

It supports `BAAI/bge-reranker-v2-minicpm-layerwise`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-minicpm-layerwise', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-minicpm-layerwise', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to('cuda')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = get_inputs(pairs, tokenizer).to(model.device)
    all_scores = model(**inputs, return_dict=True, cutoff_layers=[28])
    all_scores = [scores[:, -1].view(-1, ).float() for scores in all_scores[0]]
    print(all_scores)
```

#### 4. LLM-based lightweight reranker

It supports `BAAI/bge-reranker-v2.5-gemma2-lightweight`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def last_logit_pool(logits: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Predict whether passage B contains an answer to query A."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    query_lengths = []
    prompt_lengths = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
        query_lengths.append(len([tokenizer.bos_token_id] + query_inputs['input_ids'] + sep_inputs))
        prompt_lengths.append(len(sep_inputs + prompt_inputs))
        
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    ), query_lengths, prompt_lengths

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2.5-gemma2-lightweight', trust_remote_code=True)
tokenizer.padding_side = 'right'
model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2.5-gemma2-lightweight', trust_remote_code=True)
model = model.to('cuda')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs, query_lengths, prompt_lengths = get_inputs(pairs, tokenizer)
    inputs = inputs.to(model.device)
    outputs = model(**inputs,
                    return_dict=True,
                    cutoff_layers=[28],
                    compress_ratio=2,
                    compress_layer=[24, 40],
                    query_lengths=query_lengths,
                    prompt_lengths=prompt_lengths)
    scores = []
    for i in range(len(outputs.logits)):
        logits = last_logit_pool(outputs.logits[i], outputs.attention_masks[i])
        scores.append(logits.cpu().float().tolist())
    print(scores)
```

## Load model in local

### Load llm-based layerwise reranker in local

If you download reranker-v2-minicpm-layerwise, you can load it with the following method:

1. make sure `configuration_minicpm_reranker.py` and `modeling_minicpm_reranker.py` from [BAAI/bge-reranker-v2-minicpm-layerwise](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise) in your local path.
2. modify the following part of `config.json`:

```
"auto_map": {
    "AutoConfig": "configuration_minicpm_reranker.LayerWiseMiniCPMConfig",
    "AutoModel": "modeling_minicpm_reranker.LayerWiseMiniCPMModel",
    "AutoModelForCausalLM": "modeling_minicpm_reranker.LayerWiseMiniCPMForCausalLM"
  },
```

### Load llm-based lightweight reranker in local

1. make sure `gemma_config.py` and `gemma_model.py` from [BAAI/bge-reranker-v2.5-gemma2-lightweight](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight/tree/main) in your local path.
2. modify the following part of config.json:

```
"auto_map": {
    "AutoConfig": "gemma_config.CostWiseGemmaConfig",
    "AutoModel": "gemma_model.CostWiseGemmaModel",
    "AutoModelForCausalLM": "gemma_model.CostWiseGemmaForCausalLM"
  },
```

## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
@misc{li2023making,
      title={Making Large Language Models A Better Foundation For Dense Retrieval}, 
      author={Chaofan Li and Zheng Liu and Shitao Xiao and Yingxia Shao},
      year={2023},
      eprint={2312.15503},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{chen2024bge,
      title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation}, 
      author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2402.03216},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{li2024makingtextembeddersfewshot,
      title={Making Text Embedders Few-Shot Learners}, 
      author={Chaofan Li and MingHao Qin and Shitao Xiao and Jianlyu Chen and Kun Luo and Yingxia Shao and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2409.15700},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2409.15700}, 
}
```