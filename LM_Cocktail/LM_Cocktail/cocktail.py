import torch
import random
import numpy as np
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models

from .utils import load_model, get_model_param_list, merge_param, compute_weights


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
        tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0])
        tokenizer.save_pretrained(output_path)
        
        if model_type == "encoder":
            print(f"Transform the model to the format of 'sentence_transformers' (pooling_method='cls', normalized=True)")
            save_ckpt_for_sentence_transformers(ckpt_dir=output_path)
    return model


def mix_models_with_data(model_names_or_paths: List[str], 
                         model_type: str, 
                         example_ata: List[Dict], 
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
        example_ata (List[Any]): a list of examples
        temperature (float, optional): temperature can impact the distribution of weights . Defaults to 3.0.
        batch_size (int, optional): batch size to compute loss. Defaults to 2.
        max_input_length (int, optional): max number of input tokens for model. Defaults to 2048.
        neg_number (int, optional): the number of negatives when compute contrastive loss for embedding model. Defaults to 7.
        output_path (str, optional): path to save the mixed model. Defaults to None.

    Returns:
        new model
    """
    
    assert model_type in ['decoder', 'encoder']
    
    model = load_model(model_names_or_paths[0], model_type=model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0])
    param_list = get_model_param_list(model_names_or_paths, model_type=model_type)
    
    weights = compute_weights(model, tokenizer=tokenizer, param_list=param_list, model_type=model_type, 
                              example_data=example_ata, temperature=temperature, neg_number=neg_number,
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



