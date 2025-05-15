import math
import os.path
import random
from dataclasses import dataclass
from typing import Iterator

import datasets
from torch.utils.data import Dataset, IterableDataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import CLIPImageProcessor


from PIL import Image
import json
import torch
import torch.distributed

from io import BytesIO
import warnings

class MMIT_Dataset(Dataset):
    def __init__(self, captions, image_ids, image_dir, image_processor) -> None:
        img_id_example = image_ids[0]
        img_id_example = str(img_id_example)
        if img_id_example[-4:] in [".jpg", ".png", "JPEG"]:
            self.image_path =[os.path.join(image_dir, str(id)) for id in image_ids]
        else:
            warnings.warn("Not found file extention in image_ids, will forcefully add '.jpg'.", UserWarning)
            self.image_path =[os.path.join(image_dir, str(id) + '.jpg') for id in image_ids]
        self.captions = captions
        self.image_processor = image_processor
    
    def __getitem__(self, item):
        pil_data = Image.open(self.image_path[item])
        pil_data = pil_data.convert('RGB')
        image = self.image_processor(pil_data)
        
            

        
        caption = self.captions[item]

        return caption, image

    def __len__(self):
        return len(self.image_path)


class MMIT_Collator:
    def __init__(self, tokenizer, caption_max_len):
        self.tokenizer = tokenizer
        self.caption_max_len = caption_max_len
    


    def __call__(self, features):
        caption = [f[0] for f in features]
        images = [f[1] for f in features]
        
        c_collated = self.tokenizer(
            caption,
            truncation=True,
            padding = True,
            max_length=self.caption_max_len,
            return_tensors="pt",
        )

        # i_collated = torch.stack(images)    
        
        # for clip model
        images = [f["pixel_values"][0] for f in images]
        images = [torch.tensor(arr) for arr in images]
        i_collated = torch.stack(images)    
        ##clip_end

        return c_collated, i_collated
    
class Image_Dataset(Dataset):
    def __init__(self, image_ids, image_dir, image_processor) -> None:

        self.image_path =[os.path.join(image_dir, str(id)) for id in image_ids]
        self.image_processor = image_processor
    
    def __getitem__(self, item):
        pil_data = Image.open(self.image_path[item])
        image = self.image_processor(pil_data)

        return image

    def __len__(self):
        return len(self.image_path)

class Image_Collator:
    def __init__(self, tokenizer, caption_max_len):
        self.tokenizer = tokenizer
        self.caption_max_len = caption_max_len
    

    def __call__(self, features):
        # images = features
        # i_collated = torch.stack(images)    

        # for clip model
        images = [f["pixel_values"][0] for f in features]
        images = [torch.tensor(arr) for arr in images]
        i_collated = torch.stack(images)    
        ## clip-end
        return i_collated