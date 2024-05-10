import os
import shutil

import torch
import random
import numpy as np
from typing import List, Dict, Any

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from transformers import pipeline

from .utils import load_model, get_model_param_list, merge_param, compute_weights, get_model_param_dirs, merge_param_by_layer


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normalized: bool = True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode=pooling_mode)
    if normalized:
        normalized_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalized_layer],
                                    device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)


def mix_models(model_names_or_paths: List[str], 
               model_type: str, 
               weights: List[float], 
               output_path: str=None):
    """_summary_
    mix models based on given weights
    Args:
        model_names_or_paths (List[str]): a list of names or paths to models
        model_type (str): type of model to mix, should be in ["decoder", "encoder", "reranker"]
        weights (List[float]): a list of mixing weights. The sum of weights should be equal to 1.
        output_path (str, optional): path to save the mixed model. Defaults to None.

    Returns:
        new model
    """
    
    assert len(model_names_or_paths) == len(weights)
    assert model_type in ['decoder', 'encoder', 'reranker']
    assert sum(weights) - 1 <= 1e-3
    
    param_list = get_model_param_list(model_names_or_paths, model_type=model_type)
    new_param = merge_param(param_list, weights=weights)
    
    print("***weight for each model***: ")
    for w, n in zip(weights, model_names_or_paths):
        print(n, w)
    
    model = load_model(model_names_or_paths[0], model_type=model_type)
    model.load_state_dict(new_param)
    
    if output_path is not None:
        print(f"Saving the new model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        
        if model_type == "encoder":
            print(f"Transform the model to the format of 'sentence_transformers' (pooling_method='cls', normalized=True)")
            save_ckpt_for_sentence_transformers(ckpt_dir=output_path)
    return model


def mix_models_with_data(model_names_or_paths: List[str], 
                         model_type: str, 
                         example_data: List[Dict], 
                         temperature: float=5.0,
                         batch_size:int=2, 
                         max_input_length:int=2048, 
                         neg_number: int=7,
                         output_path: str=None):
    """_summary_
    mix model based on given a few examples
    Args:
        model_names_or_paths (List[str]):  a list of names or paths to models
        model_type (str): type of model to mix, should be in ["decoder", "encoder"]
        example_data (List[Any]): a list of examples
        temperature (float, optional): temperature can impact the distribution of weights . Defaults to 3.0.
        batch_size (int, optional): batch size to compute loss. Defaults to 2.
        max_input_length (int, optional): max number of input tokens for model. Defaults to 2048.
        neg_number (int, optional): the number of negatives when compute contrastive loss for embedding model. Defaults to 7.
        output_path (str, optional): path to save the mixed model. Defaults to None.

    Returns:
        new model
    """
    
    assert model_type in ['decoder', 'encoder', 'encoder-decoder']
    
    model = load_model(model_names_or_paths[0], model_type=model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
    param_list = get_model_param_list(model_names_or_paths, model_type=model_type)
    
    weights = compute_weights(model, tokenizer=tokenizer, param_list=param_list, model_type=model_type, 
                              example_data=example_data, temperature=temperature, neg_number=neg_number,
                              batch_size=batch_size, max_input_length=max_input_length)
    
    print("***weight for each model***: ")
    for w, n in zip(weights, model_names_or_paths):
        print(n, w)
    
    new_param = merge_param(param_list, weights=weights)    
    model.load_state_dict(new_param)
    
    if output_path is not None:
        print(f"Saving the new model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        if model_type == "encoder":
            print(f"Transform the model to the format of 'sentence_transformers' (pooling_method='cls', normalized=True)")
            save_ckpt_for_sentence_transformers(ckpt_dir=output_path)
        
    return model


def mix_models_by_layers(model_names_or_paths: List[str], 
                         model_type: str, 
                         weights: List[float], 
                         output_path: str=None):
    """_summary_
    mix models based on given weights, and load them layer by layer
    Args:
        model_names_or_paths (List[str]): a list of names or paths to models
        model_type (str): type of model to mix, should be in ["decoder", "encoder", "reranker"]
        weights (List[float]): a list of mixing weights. The sum of weights should be equal to 1.
        output_path (str, optional): path to save the mixed model. Defaults to None.

    Returns:
        new model
    """
    
    assert len(model_names_or_paths) == len(weights)
    assert model_type in ['decoder', 'encoder', 'reranker']
    assert sum(weights) - 1 <= 1e-3

    param_dirs, temp_dir = get_model_param_dirs(model_names_or_paths, model_type=model_type)
    temp_file_path = merge_param_by_layer(param_dirs, weights=weights)
    
    print("***weight for each model***: ")
    for w, n in zip(weights, model_names_or_paths):
        print(n, w)

    with init_empty_weights():
        if model_type == 'decoder':
            meta_model = AutoModelForCausalLM.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        elif model_type == 'encoder':
            meta_model = AutoModel.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        elif model_type == 'reranker':
            model = AutoModelForSequenceClassification.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        else:
            raise NotImplementedError(f"not support this model_type: {model_type}")

    device_map = {name: "cpu" for name, _ in meta_model.named_modules()}
    model = load_checkpoint_and_dispatch(meta_model, checkpoint=temp_file_path, device_map=device_map)
    model.tie_weights()

    os.remove(temp_file_path)
    shutil.rmtree(temp_dir)
    print(f"Remove temporary file: {temp_file_path}")
    print(f"Remove temporary directory: {temp_dir}")

    if output_path is not None:
        print(f"Saving the new model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0])
        tokenizer.save_pretrained(output_path)
        
        if model_type == "encoder":
            print(f"Transform the model to the format of 'sentence_transformers' (pooling_method='cls', normalized=True)")
            save_ckpt_for_sentence_transformers(ckpt_dir=output_path)
    return model
