import os
from dataclasses import dataclass, field
from transformers.training_args import TrainingArguments
from typing import Optional, List, Union


@dataclass
class BaseArgs:
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Default path to save language models.'}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Default path to save huggingface datasets.'}
    )
    data_root: str = field(
        default="/data/llm-embedder", 
        metadata={'help': 'The base directory storing all data used for training and evaluation. If specified, make sure all train_data, eval_data, and corpus are path relative to data_root!'},
    )
    train_data: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'Training json file or glob to match a list of files.'},
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={'help': 'Evaluation json file.'},
    )
    corpus: str = field(
        default=None,
        metadata={'help': 'Corpus jsonl file.'}
    )
    key_template: str = field(
        default="{title} {text}",
        metadata={'help': 'How to concatenate columns in the corpus to form one key?'}
    )
    metrics: List[str] = field(
        default_factory=lambda: ["mrr", "recall", "ndcg"],
        metadata={'help': 'List of metrics'}
    )
    cutoffs: List[int] = field(
        default_factory=lambda: [1, 5, 10, 100],
        metadata={'help': 'Cutoffs to evaluate retrieval metrics.'}
    )
    filter_answers: bool = field(
        default=False,
        metadata={'help': 'Remove negatives that contain the desired answer when collating negatives?'}
    )
    max_neg_num: int = field(
        default=100,
        metadata={'help': 'Maximum negative number to mine.'}
    )
    
    load_result: bool = field(
        default=False,
        metadata={'help': 'Load retrieval results directly?'}
    )
    save_result: bool = field(
        default=True,
        metadata={'help': 'Save retrieval results?'}
    )
    save_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Name suffix of the json file when saving the collated retrieval results.'}
    )
    save_to_output: bool = field(
        default=False,
        metadata={'help': 'Save the result/key/negative to output_dir? If not true, they will be saved next to the eval_data.'}
    )
    
    def resolve_path(self, path):
        """Resolve any path starting with 'llm-embedder:' to relative path against data_root."""
        pattern = "llm-embedder:"
        # resolve relative data paths when necessary
        if isinstance(path, list):
            for i, x in enumerate(path):
                if x.startswith(pattern):
                    path[i] = os.path.join(self.data_root, x.replace(pattern, ""))
        else:
            if path.startswith(pattern):
                path = os.path.join(self.data_root, path.replace(pattern, ""))

        return path

    def __post_init__(self):        
        if self.train_data is not None:
            self.train_data = self.resolve_path(self.train_data)

        if self.eval_data is not None:
            self.eval_data = self.resolve_path(self.eval_data)

        if self.corpus is not None:
            self.corpus = self.resolve_path(self.corpus)


@dataclass
class DenseRetrievalArgs(BaseArgs):
    query_encoder: str = field(
        default="BAAI/bge-base-en",
        metadata={'help': 'Path to encoder model or model identifier from huggingface.co/models.'}
    )
    key_encoder: str = field(
        default="BAAI/bge-base-en",
        metadata={'help': 'Path to encoder model or model identifier from huggingface.co/models.'}
    )
    add_instruction: bool = field(
        default=True,
        metadata={'help': 'Add instruction for each task?'}
    )
    version: str = field(
        default="bge",
        metadata={'help': 'Version for configs.'}
    )
    query_max_length: int = field(
        default=256,
        metadata={'help': 'Max query length.'}
    )
    key_max_length: int = field(
        default=256,
        metadata={'help': 'Max key length.'}
    )
    truncation_side: str = field(
        default="right",
        metadata={'help': 'Which side to truncate?'}
    )

    pooling_method: List[str] = field(
        default_factory=lambda: ["cls"],
        metadata={'help': 'Pooling methods to aggregate token embeddings for a sequence embedding. {cls, mean, dense, decoder}'}
    )
    tie_encoders: bool = field(
        default=True,
        metadata={'help': 'Tie query encoder and key encoder? If True, then the query_encoder_name is used.'}
    )

    dense_metric: str = field(
        default="cos",
        metadata={'help': 'What type of metric for dense retrieval? ip, l2, or cos.'}
    )
    faiss_index_factory: str = field(
        default="Flat",
        metadata={'help': 'Index factory string for faiss.'}
    )
    hits: int = field(
        default=200,
        metadata={'help': 'How many keys to retrieve?'}
    )
    batch_size: int = field(
        default=1000,
        metadata={'help': 'Batch size for indexing and retrieval.'}
    )

    load_encode: bool = field(
        default=False,
        metadata={'help': 'Load cached embeddings?'}
    )
    save_encode: bool = field(
        default=False,
        metadata={'help': 'Save embeddings?'}
    )
    load_index: bool = field(
        default=False,
        metadata={'help': 'Load cached index?'}
    )
    save_index: bool = field(
        default=False,
        metadata={'help': 'Save index?'}
    )
    embedding_name: str = field(
        default="embeddings",
        metadata={'help': 'The embedding name for saving? (Also used for faiss index name.)'}
    )
    dtype: str = field(
        default="fp16",
        metadata={'help': 'Data type for retriever.'}
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )


@dataclass
class BM25Args(BaseArgs):
    anserini_dir: str = field(
        default='/share/peitian/Apps/anserini',
        metadata={'help': 'Anserini installation directory.'}
    )

    k1: float = field(
        default=0.82,
        metadata={'help': 'BM25 k1.'}
    )
    b: float = field(
        default=0.68,
        metadata={'help': 'BM25 b.'}
    )
    storeDocvectors: bool = field(
        default=False,
        metadata={'help': 'Store document vector? Useful when you want to inspect the word-level statistics (tf-idf) after index construction.'}
    )
    hits: int = field(
        default=200,
        metadata={'help': 'How many keys to retrieve?'}
    )
    language: str = field(
        default="en",
        metadata={'help': 'Language.'}
    )
    threads: int = field(
        default=32,
        metadata={'help': 'Indexing/Searching thread number.'}
    )
    load_index: bool = field(
        default=False,
        metadata={'help': 'Load index?'}
    )
    load_collection: bool = field(
        default=False,
        metadata={'help': 'Load collection?'}
    )


@dataclass
class RankerArgs(BaseArgs):
    ranker: str = field(
        default="BAAI/bge-base-en",
        metadata={'help': 'Ranker name or path.'}
    )
    ranker_method: str = field(
        default="cross-encoder",
        metadata={'help': 'What kind of ranker to use? {cross: cross encoder}'}
    )
    dtype: str = field(
        default="fp16",
        metadata={'help': 'Data type for ranker.'}
    )

    query_max_length: int = field(
        default=256,
        metadata={'help': 'Max query length.'}
    )
    key_max_length: int = field(
        default=256,
        metadata={'help': 'Max key length.'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add instruction for each task?'}
    )
    version: str = field(
        default="bge",
        metadata={'help': 'Version for configs.'}
    )

    hits: Optional[int] = field(
        default=None,
        metadata={'help': 'How many top reranked keys to keep?'}
    )
    batch_size: int = field(
        default=4,
        metadata={'help': 'Batch size for indexing and retrieval.'}
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )


@dataclass
class RetrievalArgs(DenseRetrievalArgs, BM25Args):
    retrieval_method: str = field(
        default="dense",
        metadata={'help': 'How to retrieve? {dense, bm25, random, no}'}
    )


@dataclass
class RetrievalTrainingArgs(TrainingArguments):
    output_dir: str = field(
        default='data/outputs/',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'},
    )
    eval_method: str = field(
        default="retrieval",
        metadata={'help': 'How to evaluate?'},
    )

    use_train_config: bool = field(
        default=False,
        metadata={'help': 'Use training config from TASK_CONFIG to override arguments?'}
    )
    inbatch_same_dataset: Optional[str] = field(
        default=None,
        metadata={'help': 'Whether and how to use samples from the same task in each batch (across devices). {epoch, random}'}
    )
    negative_cross_device: bool = field(
        default=True,
        metadata={'help': 'Gather negatives from all devices when distributed training?'}
    )
    cos_temperature: float = field(
        default=0.01,
        metadata={'help': 'Temperature used for cosine dense metric.'}
    )
    teacher_temperature:float = field(
        default=1.,
        metadata={'help': 'Temperature used for cosine dense metric.'}
    )
    student_temperature:float = field(
        default=1.,
        metadata={'help': 'Temperature used for cosine dense metric.'}
    )
    contrastive_weight: float = field(
        default=0.2,
        metadata={'help': 'Weight for contrastive loss.'}
    )
    distill_weight: float = field(
        default=1.0,
        metadata={'help': 'Weight for distillation loss.'}
    )
    stable_distill: bool = field(
        default=False, 
        metadata={'help': 'Sort distillation.'}
    )

    max_sample_num: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples at most for training dataset?'}
    )
    train_group_size: int = field(
        default=8,
        metadata={'help': 'How many keys in a batch?'}
    )
    select_positive: str = field(
        default="first",
        metadata={'help': 'How to select the positive key from a set of positives?'}
    )
    select_negative: str = field(
        default="random",
        metadata={'help': 'How to select the negative keys from a set of negatives?'}
    )
    teacher_scores_margin: Optional[float] = field(
        default=None,
        metadata={'help': 'Minimum margin in teacher_scores. The samples with smaller margin will be removed from training.'}
    )
    teacher_scores_min: Optional[float] = field(
        default=None,
        metadata={'help': 'Minimum teacher_scores. The samples whose biggest score is lower than this will be removed from training.'}
    )

    per_device_train_batch_size: int = field(
        default=16,
        metadata={'help': 'Train batch size'},
    )
    learning_rate: float = field(
        default=5e-6,
        metadata={'help': 'Learning rate.'},
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={'help': 'Warmup ratio for linear scheduler.'},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={'help': 'Weight decay in AdamW.'},
    )

    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 training?'}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={'help': 'Find unused parameters in torch DDP?'},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Remove columns that are not registered in the forward function of the model?'},
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={'help': 'Evaluation strategy'},
    )
    save_steps: int = field(
        default=2000,
        metadata={'help': 'Saving frequency.'},
    )
    logging_steps: int = field(
        default=100,
        metadata={'help': 'Logging frequency according to logging strategy.'},
    )
    early_exit_steps: Optional[int] = field(
        default=None,
        metadata={'help': 'After how many steps to exit training loop.'},
    )

    report_to: str = field(
        default="none", metadata={"help": "The list of integrations to report the results and logs to."}
    )
    log_path: str = field(
        default="data/results/performance.log",
        metadata={'help': 'Pooling method to aggregate token embeddings for a sequence embedding.'}
    )
    
    # NOTE: newer version of transformers forbid modifying the configs after initilization, we bypass this setting
    def __setattr__(self, name, value):
        super(TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        # for convenience
        # self.eval_steps = self.save_steps

