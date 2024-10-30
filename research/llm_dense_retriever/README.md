<div align="center">
<h1> BGE-ICL </h1>
</div>

**BGE-EN-ICL** primarily demonstrates the following capabilities:
- In-context learning ability: By providing few-shot examples in the query, it can significantly enhance the model's ability to handle new tasks.
- Outstanding performance: The model has achieved state-of-the-art (SOTA) performance on both BEIR and AIR-Bench.

## 📑 Open-source Plan

- [x] Checkpoint
- [x] Training Data
- [x] Training Code
- [x] Technical Report
- [ ] Evaluation Pipeline

The technical report for **BGE-EN-ICL** can be found in [Making Text Embedders Few-Shot Learners](https://arxiv.org/abs/2409.15700)

## Environment
```bash
conda create icl python=3.10

conda activate icl

# You may need to adjust the cuda version
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.41.0 deepspeed accelerate datasets peft pandas
pip install flash-attn --no-build-isolation
```

## Model List

| Model                                                        | Introduction                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BAAI/bge-en-icl](https://huggingface.co/BAAI/bge-en-icl) | BGE-ICL trained on the full dataset |
| BAAI/bge-en-icl-e5data | BGE-ICL trained on the same public dataset as e5-mistral |

## Data List

| Data                                                        | Introduction                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [public-data](https://huggingface.co/datasets/cfli/bge-e5data) | Public data identical to [e5-mistral](https://huggingface.co/intfloat/e5-mistral-7b-instruct) |
| [full-data](https://huggingface.co/datasets/cfli/bge-full-data) | The full dataset we used for training |

## Usage 

### Using FlagEmbedding
```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install -e .
```

```python
from FlagEmbedding import FlagICLModel
queries = ["how much protein should a female eat", "summit define"]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
examples = [
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'what is a virtual interface',
   'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."},
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'causes of back pain in female for a week',
   'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."}
]
model = FlagICLModel('BAAI/bge-en-icl', 
                     query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query.",
                     examples_for_task=examples,  # set `examples_for_task=None` to use model without examples
                     use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode_queries(queries)
embeddings_2 = model.encode_corpus(documents)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

By default, FlagICLModel will use all available GPUs when encoding. Please set `os.environ["CUDA_VISIBLE_DEVICES"]` to select specific GPUs.
You also can set `os.environ["CUDA_VISIBLE_DEVICES"]=""` to make all GPUs unavailable.


### Using HuggingFace Transformers

With the transformers package, you can use the model like this: First, you pass your input through the transformer model, then you select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.

```python
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'

def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'

def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
            tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
    return new_max_length, new_queries

task = 'Given a web search query, retrieve relevant passages that answer the query.'
examples = [
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'what is a virtual interface',
   'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."},
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'causes of back pain in female for a week',
   'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."}
]
examples = [get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
examples_prefix = '\n\n'.join(examples) + '\n\n' # if there not exists any examples, just set examples_prefix = ''
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, 'summit define')
]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
query_max_len, doc_max_len = 512, 512

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-en-icl')
model = AutoModel.from_pretrained('BAAI/bge-en-icl')
model.eval()

new_query_max_len, new_queries = get_new_queries(queries, query_max_len, examples_prefix, tokenizer)

query_batch_dict = tokenizer(new_queries, max_length=new_query_max_len, padding=True, truncation=True, return_tensors='pt')
doc_batch_dict = tokenizer(documents, max_length=doc_max_len, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    query_outputs = model(**query_batch_dict)
    query_embeddings = last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    doc_outputs = model(**doc_batch_dict)
    doc_embeddings = last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])
    
# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
scores = (query_embeddings @ doc_embeddings.T) * 100
print(scores.tolist())
```

## Fine-tune

Here is an example for fine-tune:
```shell
cd ./finetune
torchrun --nproc_per_node 8 \
run.py \
--output_dir ./test \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--train_data cfli/bge-e5data \
--learning_rate 1e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 16 \
--lora_alpha 64 \
--lora_rank 32 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 2048 \
--passage_max_len 512 \
--example_query_max_len 256 \
--example_passage_max_len 256 \
--train_group_size 8 \
--logging_steps 1 \
--save_steps 250 \
--save_total_limit 20 \
--ddp_find_unused_parameters False \
--negatives_cross_device \
--gradient_checkpointing \
--deepspeed ../../LLARA/stage1.json \
--warmup_steps 100 \
--fp16 \
--cache_dir ./cache/model_cache \
--token ... \
--cache_path ./cache/data_cache \
--sub_batch_size 64 \
--target_modules q_proj k_proj v_proj o_proj down_proj up_proj gate_proj \
--use_special_tokens \
--symmetric_batch_size 256 \
--symmetric_train_group_size 8 \
--max_class_neg 7 \
--save_merged_lora_model True
```

## Citation

If you find this repository useful, please give us a star ⭐.

To cite our work:

```
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
