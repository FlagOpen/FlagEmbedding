import torch
import random
import numpy as np
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification


def load_llm(model_name:str, trust_remote_code:bool):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"})
    return model


def load_embedder(model_name:str, trust_remote_code:bool):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"})
    return model


def load_reranker(model_name:str, trust_remote_code:bool):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code, device_map = {"": "cpu"})
    return model


def load_model(model_name:str, model_type:str, trust_remote_code:bool=True):
    if model_type == 'decoder':
        model = load_llm(model_name, trust_remote_code=trust_remote_code)
    elif model_type == 'encoder':
        model = load_embedder(model_name, trust_remote_code=trust_remote_code)
    elif model_type == 'reranker':
        model = load_reranker(model_name, trust_remote_code=trust_remote_code)
    else:
        raise NotImplementedError(f"not support this model_type: {model_type}")
    return model


def get_model_param_list(model_names: List[str], model_type:str):
    model_param_list = []
    for name in model_names:
        print(f"loading {name} -----------------")
        model = load_model(name, model_type=model_type)
        model_param_list.append(model.state_dict())
    return model_param_list


def merge_param(model_param_list: List[Dict], weights: List[float]):
    new_param = {}
    for k in model_param_list[0].keys():
        for w, param in zip(weights, model_param_list):
            if param[k].dtype == torch.int64 or param[k].dtype == torch.int32:
                new_param[k] = param[k]
            elif k not in new_param:
                new_param[k] = w * param[k]
            else:
                new_param[k] += w * param[k]
    return new_param



def compute_weights(base_model, tokenizer, param_list: List[Dict], model_type: str, example_data: List[Any], temperature: float=5.0, batch_size:int=2, max_input_length:int=2048, neg_number:int=7):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    base_model = base_model.to(device)
    
    if model_type == 'decoder':
        input_data = preprocess_data_for_llm(example_data=example_data, tokenizer=tokenizer, device=device, batch_size=batch_size, max_input_length=max_input_length)
        loss_func = llm_loss
    elif model_type == 'encoder':
        input_data = preprocess_data_for_embedder(example_data=example_data, tokenizer=tokenizer, device=device, batch_size=batch_size, max_input_length=max_input_length, neg_number=neg_number)
        loss_func = embedder_loss

    example_loss = [] 
    with torch.no_grad():
        for params in param_list:
            base_model.load_state_dict(params)
            loss = loss_func(base_model=base_model, input_data=input_data)
            example_loss.append(loss)
    
    weights = torch.softmax(-torch.FloatTensor(example_loss)/temperature, -1).numpy().tolist()
    return weights




def preprocess_data_for_embedder(example_data, tokenizer, device, batch_size:int=64, max_input_length:int=512, neg_number:int=7):
    input_data = []
    quries = []
    passages = []
    # max_input_length = min(512, max_input_length)
    for e in example_data:
        quries.append(e['query'])
        passages.append(e['pos'][0])
        passages.extend(random.sample(e['neg'], neg_number))

        if len(quries) == batch_size:
            q_tokens = tokenizer(quries, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
            p_tokens =  tokenizer(passages, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
            q_tokens, p_tokens = q_tokens.to(device), p_tokens.to(device)
            input_data.append([q_tokens, p_tokens])
    if len(quries) > 0:
        q_tokens = tokenizer(quries, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
        p_tokens =  tokenizer(passages, padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
        q_tokens, p_tokens = q_tokens.to(device), p_tokens.to(device)
        input_data.append([q_tokens, p_tokens])
        
    return input_data


def embedder_loss(base_model, input_data):
    def generate_embeddings(model, inputs):
        embeddings = model(**inputs, return_dict=True).last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)        
        return embeddings

    with torch.no_grad():
        loss = 0
        for q_inputs, p_inputs in input_data:
            q_embeddings = generate_embeddings(base_model, q_inputs)
            p_embeddings = generate_embeddings(base_model, p_inputs)
            scores = torch.matmul(q_embeddings, p_embeddings.transpose(0, 1)) / 0.05
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_embeddings.size(0) // q_embeddings.size(0))
            batch_loss = torch.nn.CrossEntropyLoss(reduction='mean')(scores, target)
            loss += batch_loss.cpu()
    loss = float(loss / len(input_data))
    return float(loss)


def preprocess_data_for_llm(example_data, tokenizer, device, batch_size:int=2, max_input_length:int=2048):
    batch_input_ids = []
    batch_labels = []
    batch_max_length = max_input_length
    for data in example_data:
        input, output = data['input'], data['output']
            
        input_ids = tokenizer.encode(input+' '+output)
        input_ids.append(tokenizer.eos_token_id)
            
        prompt_ids = tokenizer.encode(input)
        labels = [-100]*len(prompt_ids) + input_ids[len(prompt_ids):]
    
        input_ids = input_ids[:batch_max_length]
        input_ids += [tokenizer.pad_token_id] * (batch_max_length - len(input_ids))
        batch_input_ids.append(input_ids)
        
        labels = labels[:batch_max_length]
        labels += [-100] * (batch_max_length - len(labels))
        batch_labels.append(labels)
        
    batch_input_ids = torch.LongTensor(batch_input_ids).to(device)
    batch_labels = torch.LongTensor(batch_labels).to(device)
    attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).to(device)
    
    batch_data = []
    for i in range(0, len(batch_input_ids), batch_size):
        batch_data.append(dict(
            input_ids=batch_input_ids[i:i+batch_size],
            labels=batch_labels[i:i+batch_size],
            attention_mask=attention_mask[i:i+batch_size],
            ))
    return batch_data



def llm_loss(base_model, input_data):
    loss = 0
    with torch.no_grad():
        for data in input_data:
            output = base_model(**data)
            loss += output.loss.cpu()
    loss = float(loss / len(input_data))
    return loss




