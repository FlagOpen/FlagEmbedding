# Activation-Beacon

This folder contains the newer code for activation beacon with the support of **Mistral models**, **Deepspeed Zero3 training**, **chat templates**, and **more evaluation tasks**. The code here are under development and subject to change in the future.

## Environment
```bash
conda create beacon python=3.10.14

conda activate beacon

conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.39.3 deepspeed accelerate datasets peft pandas seaborn rouge fuzzywuzzy jieba
pip install flash-attn --no-build-isolation
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


## Data
You should download the data for fine-tuning & evaluation then untar the file at anywhere you prefer, e.g. `/data`:
```bash
# feel free to alternate /data to your prefered location
wget https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/activation-beacon-new.tar.gz?download=true -O /data/activation-beacon-new.tar.gz

cd /data
tar -xzvf activation-beacon-new.tar.gz
```

**IMPORTANT NOTE**

For any path specified for `train_data` and `eval_data`: if it is prefixed with `activation-beacon:`, it will be solved to the relative path against [`data_root`](./src/args.py). 
  - e.g. `activation-beacon:lm/pg19.json` becomes `${data_root}/lm/pg19.json`
  - you can modify the default value of [`data_root`](./src/args.py), so that you don't need to type it for each command.


## Training
See [training section](./docs/training.md).

## Evaluation
See [evaluation section](./docs/evaluation.md). 


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