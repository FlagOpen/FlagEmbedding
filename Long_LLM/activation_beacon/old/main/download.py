from src.args import ModelArgs
from datasets import load_dataset, get_dataset_config_names
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, HfArgumentParser
from typing import Optional, List


@dataclass
class DownloadArgs(ModelArgs):
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    use_lm_class: bool = field(
        default=False,
        metadata={'help': 'Call .from_pretrained from AutoModelForCausalLM? Useful when downloading remote-code based lms.'}
    )
    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Dataset name'}
    )
    dataset_subset: Optional[str] = field(
        default=None,
        metadata={'help': 'Dataset subset name'}
    )
    dataset_split: Optional[str] = field(
        default=None,
        metadata={'help': 'Dataset split'}
    )
    
    revision: str = field(
        default=None,
        metadata={'help': 'Remote code revision'}
    )
    resume_download: bool = field(
        default=False,
        metadata={'help': 'Resume downloading'}
    )
    def __post_init__(self):
        # folder or model not exists
        if self.model_name_or_path is not None:
            kwargs = {
                'revision': self.revision,
                'resume_download': self.resume_download
            }
            AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.model_cache_dir, trust_remote_code=True, **kwargs)
            if self.use_lm_class:
                AutoModelForCausalLM.from_pretrained(self.model_name_or_path, cache_dir=self.model_cache_dir, trust_remote_code=True, **kwargs)
            else:
                AutoModel.from_pretrained(self.model_name_or_path, cache_dir=self.model_cache_dir, trust_remote_code=True, **kwargs)
        if self.dataset_name_or_path is not None:
            if self.dataset_subset is None:
                dataset_subsets = get_dataset_config_names(self.dataset_name_or_path)
                for dataset_subset in dataset_subsets:
                    load_dataset(self.dataset_name_or_path, name=dataset_subset, split=self.dataset_split, cache_dir=self.dataset_cache_dir)
            else:
                load_dataset(self.dataset_name_or_path, name=self.dataset_subset, split=self.dataset_split, cache_dir=self.dataset_cache_dir)


if __name__ == "__main__":
    parser = HfArgumentParser([DownloadArgs])
    args, = parser.parse_args_into_dataclasses()
