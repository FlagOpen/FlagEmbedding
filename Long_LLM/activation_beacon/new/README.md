# Activation-Beacon

This folder contains the newer code for activation beacon with the support of **Mistral models**, **Deepspeed Zero3 training**, **chat templates**, and **more evaluation tasks**. The code here are under development and subject to change in the future.

## Environment
```bash
conda create beacon python=3.10.14

conda activate beacon

conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.39.3 deepspeed accelerate datasets peft pandas seaborn
pip install flash-attn --no-build-isolation

# these packages are used in evaluation
pip install rouge fuzzywuzzy jieba
```

## Usage
```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "namespace-Pt/activation-beacon-mistral-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)

model = model.cuda().eval()

with torch.no_grad():
  # short context
  messages = [{"role": "user", "content": "Tell me about yourself."}]
  inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=50)
  print(f"Input Length: {inputs['input_ids'].shape[1]}")
  print(f"Output:       {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

  # reset memory before new generation task
  model.memory.reset()

  # long context
  with open("data/toy/infbench.json", encoding="utf-8") as f:
    example = json.load(f)
  messages = [{"role": "user", "content": example["context"]}]
  inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
  outputs = model.generate(**inputs, do_sample=False, top_p=1, temperature=1, max_new_tokens=20)[:, inputs["input_ids"].shape[1]:]
  print("*"*20)
  print(f"Input Length: {inputs['input_ids'].shape[1]}")
  print(f"Answers:      {example['answer']}")
  print(f"Prediction:   {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```
**NOTE**: It's okay to see warnings like `This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (32768). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.` Just ignore it.

## Training
See [training section](./docs/training.md). **The training script for Mistral will be released in future.**

## Evaluation
See [evaluation section](./docs/evaluation.md). 

The performance of [activation-beacon-mistral-7b](https://huggingface.co/namespace-Pt/activation-beacon-mistral-7b) is shown below.

- [Needle in a Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack):
We evaluate the model on the Needle-In-A-HayStack task using the official setting.
<img src="imgs/needle.png"></img>


- [Longbench](https://arxiv.org/abs/2308.14508): We evaluate the model on LongBench using 32K context length.

    |Model|Single Doc QA|Multi Doc QA|Summarization|
    |:-:|:-:|:-:|:-:|
    |[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)|32.70|25.87|27.42|
    |[Yarn-Mistral-128K](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)|33.71|36.08|23.47|
    |Activation-Beacon-Mistral-7B|39.14|43.27|29.52|

- [InfiniteBench](https://arxiv.org/pdf/2402.13718.pdf): We evaluate the model on InfiniteBench using 128K context length. The results of Yarn-Mistral-128K is copied from the [paper](https://arxiv.org/pdf/2402.13718.pdf).

    |Model|LongBookQA Eng|LongBookSum Eng|
    |:-:|:-:|:-:|
    |[Yarn-Mistral-128K](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)|9.55|9.09|
    |Activation-Beacon-Mistral-7B|26.81|12.49|

- [Topic Retrieval](https://lmsys.org/blog/2023-06-29-longchat/): We evaluate the model on Topic Retrieval task with `[5,10,15,20,25,30,40,50,60,70]` topics.
<img src="imgs/topic.png"></img>

- [PG19 Perplexity](https://arxiv.org/abs/2309.12307): We evaluate the sliding window perplexity on PG19 test set with window size 100K and stride 32K. We also report the latency and the GPU memory usage. For full-attention models, we enable [flash-attention-2](https://github.com/Dao-AILab/flash-attention) and [tensor parallel](https://github.com/BlackSamorez/tensor_parallel). The evaluation is run on 8xA800 machine.

    |Model|Perplexity|Latency (s)|Memory (GB)|
    |:-:|:-:|:-:|:-:|
    |[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)|8.83|14.02|525.6 (cannot run on a single GPU)|
    |[Yarn-Mistral-128K](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)|7.66|14.56|525.6 (cannot run on a single GPU)|
    |Activation-Beacon-Mistral-7B|8.16|3.06|27.4|

- [Passkey Retrieval](https://arxiv.org/abs/2309.12307): We evaluate the model on Passkey Retrieval task using the official setting.
<img src="imgs/passkey.png"></img>



## Citation
If you find this repository useful, please give us a star ⭐.

To cite our work:
```
@misc{zhang2024soaring,
    title={Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon}, 
    author={Peitian Zhang and Zheng Liu and Shitao Xiao and Ninglu Shao and Qiwei Ye and Zhicheng Dou},
    year={2024},
    eprint={2401.03462},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```