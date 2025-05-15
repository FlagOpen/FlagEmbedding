import os
import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from flag_mmret import Flag_mmret
import json

logger = logging.getLogger(__name__)


@dataclass
class Args:
    model_name: str = field(
        default="BAAI/BGE-VL-large",
        metadata={'help': 'Model Name'}
    )
    result_save_path: str = field(
        default="./eval/mmret_large_circo.json",
        metadata={'help': 'Where to save the results.'}
    )
    image_dir: str = field(
        default="YOUR_COCO_IMAGE_DIRECTORY",
        metadata={'help': 'Where the images located on.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    max_query_length: int = field(
        default=64,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=77,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )
    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )



def index(model: Flag_mmret, corpus: datasets.Dataset, batch_size: int = 256, max_length: int=512, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    
    else:
        
        corpus_embeddings = model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length, corpus_type='image')
                
        dim = corpus_embeddings.shape[-1]
        
        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    
    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
    

    if model.device == torch.device("cuda"):
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)

   

    return faiss_index


def search(model: Flag_mmret, queries: datasets, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode_queries([queries["q_text"], queries["q_img"]], 
                                            batch_size=batch_size, 
                                            max_length=max_length,
                                            query_type='mm_it')

    
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    

def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    
    print(f"Results will be saved in {args.result_save_path}")
    eval_data = datasets.load_dataset('json', data_files="./eval/data/circo_query.jsonl", split='train')
    image_corpus_test = datasets.load_dataset('json',  data_files="./eval/data/circo_corpus.jsonl", split='train')
    
    model = Flag_mmret(model_name=args.model_name,
                        normlized = True,
                        image_dir=args.image_dir,
                        use_fp16=False,
                    )
    
    
    faiss_index = index(
        model=model, 
        corpus=image_corpus_test, 
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )
    
    scores, indices = search(
        model=model, 
        queries=eval_data, 
        faiss_index=faiss_index, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    
    
    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(image_corpus_test[indice]["content"])

    ########## results in test corpus #########
    q_images = eval_data["q_img"]

    q_ids = []
    for _img in q_images:
        _id = os.path.basename(_img)
        _id = os.path.splitext(_id)[0]
        q_ids.append(_id)
    
    pairids = eval_data["id"]
    results = {}
    for pairid, re_results, q_img in zip(pairids, retrieval_results, q_images):
        id = str(pairid)
        top_50_results = re_results[0:50]
        results[id] = top_50_results

    with open(args.result_save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()