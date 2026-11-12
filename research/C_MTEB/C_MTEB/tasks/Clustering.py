from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

NUM_SAMPLES = 2048


class CLSClusteringFastS2S(AbsTaskClusteringFast):
    max_document_to_embed = NUM_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="CLSClusteringS2S.v2",
        description="Clustering of titles from CLS dataset. Clustering of 13 sets on the main category.",
        reference="https://arxiv.org/abs/2209.05034",
        dataset={
            "path": "C-MTEB/CLSClusteringS2S",
            "revision": "e458b3f5414b62b7f9f83499ac1f5497ae2e869f",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2022-01-01", "2022-09-12"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{li2022csl,
            title={CSL: A Large-scale Chinese Scientific Literature Dataset}, 
            author={Yudong Li and Yuqing Zhang and Zhe Zhao and Linlin Shen and Weijie Liu and Weiquan Mao and Hui Zhang},
            year={2022},
            eprint={2209.05034},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }""",
        descriptive_stats={
            "n_samples": {"test": NUM_SAMPLES},
            "avg_character_length": {},
        },
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )

            check_label_distribution(self.dataset[split])

            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=NUM_SAMPLES,
        )


class CLSClusteringFastP2P(AbsTaskClusteringFast):
    max_document_to_embed = NUM_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="CLSClusteringP2P.v2",
        description="Clustering of titles + abstract from CLS dataset. Clustering of 13 sets on the main category.",
        reference="https://arxiv.org/abs/2209.05034",
        dataset={
            "path": "C-MTEB/CLSClusteringP2P",
            "revision": "4b6227591c6c1a73bc76b1055f3b7f3588e72476",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2022-01-01", "2022-09-12"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{li2022csl,
            title={CSL: A Large-scale Chinese Scientific Literature Dataset}, 
            author={Yudong Li and Yuqing Zhang and Zhe Zhao and Linlin Shen and Weijie Liu and Weiquan Mao and Hui Zhang},
            year={2022},
            eprint={2209.05034},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }""",
        descriptive_stats={
            "n_samples": {"test": NUM_SAMPLES},
            "avg_character_length": {},
        },
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )

            check_label_distribution(self.dataset[split])

            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=NUM_SAMPLES,
        )


class CLSClusteringS2S(AbsTaskClustering):
    superseded_by = "CLSClusteringS2S.v2"
    metadata = TaskMetadata(
        name="CLSClusteringS2S",
        description="Clustering of titles from CLS dataset. Clustering of 13 sets on the main category.",
        reference="https://arxiv.org/abs/2209.05034",
        dataset={
            "path": "C-MTEB/CLSClusteringS2S",
            "revision": "e458b3f5414b62b7f9f83499ac1f5497ae2e869f",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""
@article{li2022csl,
  title={CSL: A large-scale Chinese scientific literature dataset},
  author={Li, Yudong and Zhang, Yuqing and Zhao, Zhe and Shen, Linlin and Liu, Weijie and Mao, Weiquan and Zhang, Hui},
  journal={arXiv preprint arXiv:2209.05034},
  year={2022}
}
""",
        descriptive_stats={"n_samples": {"test": 100000}, "avg_character_length": None},
    )


class CLSClusteringP2P(AbsTaskClustering):
    superseded_by = "CLSClusteringP2P.v2"
    metadata = TaskMetadata(
        name="CLSClusteringP2P",
        description="Clustering of titles + abstract from CLS dataset. Clustering of 13 sets on the main category.",
        reference="https://arxiv.org/abs/2209.05034",
        dataset={
            "path": "C-MTEB/CLSClusteringP2P",
            "revision": "4b6227591c6c1a73bc76b1055f3b7f3588e72476",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@article{li2022csl,
  title={CSL: A large-scale Chinese scientific literature dataset},
  author={Li, Yudong and Zhang, Yuqing and Zhao, Zhe and Shen, Linlin and Liu, Weijie and Mao, Weiquan and Zhang, Hui},
  journal={arXiv preprint arXiv:2209.05034},
  year={2022}
}""",
        descriptive_stats={"n_samples": {"test": 100000}, "avg_character_length": None},
    )


class ThuNewsClusteringFastS2S(AbsTaskClusteringFast):
    max_document_to_embed = NUM_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="ThuNewsClusteringS2S.v2",
        dataset={
            "path": "C-MTEB/ThuNewsClusteringS2S",
            "revision": "8a8b2caeda43f39e13c4bc5bea0f8a667896e10d",
        },
        description="Clustering of titles from the THUCNews dataset",
        reference="http://thuctc.thunlp.org/",
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2006-01-01", "2007-01-01"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@software{THUCTC,
  author = {Sun, M. and Li, J. and Guo, Z. and Yu, Z. and Zheng, Y. and Si, X. and Liu, Z.},
  title = {THUCTC: An Efficient Chinese Text Classifier},
  year = {2016},
  note = {THU Chinese Text Classification Toolkit},
  publisher = {THU Natural Language Processing Lab},
  url = {https://github.com/thunlp/THUCTC}
}""",
        descriptive_stats={
            "n_samples": {"test": NUM_SAMPLES},
            "avg_character_length": {},
        },
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )

            check_label_distribution(self.dataset[split])

            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=NUM_SAMPLES,
        )


class ThuNewsClusteringFastP2P(AbsTaskClusteringFast):
    max_document_to_embed = NUM_SAMPLES
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="ThuNewsClusteringP2P.v2",
        dataset={
            "path": "C-MTEB/ThuNewsClusteringP2P",
            "revision": "5798586b105c0434e4f0fe5e767abe619442cf93",
        },
        description="Clustering of titles + abstracts from the THUCNews dataset",
        reference="http://thuctc.thunlp.org/",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=("2006-01-01", "2007-01-01"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering", "Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@software{THUCTC,
  author = {Sun, M. and Li, J. and Guo, Z. and Yu, Z. and Zheng, Y. and Si, X. and Liu, Z.},
  title = {THUCTC: An Efficient Chinese Text Classifier},
  year = {2016},
  note = {THU Chinese Text Classification Toolkit},
  publisher = {THU Natural Language Processing Lab},
  url = {https://github.com/thunlp/THUCTC}
}""",
        descriptive_stats={
            "n_samples": {"test": NUM_SAMPLES},
            "avg_character_length": {},
        },
    )

    def dataset_transform(self):
        ds = {}
        for split in self.metadata.eval_splits:
            labels = list(itertools.chain.from_iterable(self.dataset[split]["labels"]))
            sentences = list(
                itertools.chain.from_iterable(self.dataset[split]["sentences"])
            )

            check_label_distribution(self.dataset[split])

            ds[split] = Dataset.from_dict({"labels": labels, "sentences": sentences})
        self.dataset = DatasetDict(ds)
        self.dataset = self.stratified_subsampling(
            self.dataset,
            self.seed,
            self.metadata.eval_splits,
            label="labels",
            n_samples=NUM_SAMPLES,
        )


class ThuNewsClusteringS2S(AbsTaskClustering):
    superseded_by = "ThuNewsClusteringS2S.v2"
    metadata = TaskMetadata(
        name="ThuNewsClusteringS2S",
        dataset={
            "path": "C-MTEB/ThuNewsClusteringS2S",
            "revision": "8a8b2caeda43f39e13c4bc5bea0f8a667896e10d",
        },
        description="Clustering of titles from the THUCNews dataset",
        reference="http://thuctc.thunlp.org/",
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""
@inproceedings{eisner2007proceedings,
  title={Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL)},
  author={Eisner, Jason},
  booktitle={Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL)},
  year={2007}
}
@inproceedings{li2006comparison,
  title={A comparison and semi-quantitative analysis of words and character-bigrams as features in chinese text categorization},
  author={Li, Jingyang and Sun, Maosong and Zhang, Xian},
  booktitle={proceedings of the 21st international conference on computational linguistics and 44th annual meeting of the association for computational linguistics},
  pages={545--552},
  year={2006}
}
""",
        descriptive_stats={"n_samples": {"test": 100000}, "avg_character_length": None},
    )


class ThuNewsClusteringP2P(AbsTaskClustering):
    superseded_by = "ThuNewsClusteringP2P.v2"
    metadata = TaskMetadata(
        name="ThuNewsClusteringP2P",
        dataset={
            "path": "C-MTEB/ThuNewsClusteringP2P",
            "revision": "5798586b105c0434e4f0fe5e767abe619442cf93",
        },
        description="Clustering of titles + abstracts from the THUCNews dataset",
        reference="http://thuctc.thunlp.org/",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""
@inproceedings{eisner2007proceedings,
  title={Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL)},
  author={Eisner, Jason},
  booktitle={Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL)},
  year={2007}
}
@inproceedings{li2006comparison,
  title={A comparison and semi-quantitative analysis of words and character-bigrams as features in chinese text categorization},
  author={Li, Jingyang and Sun, Maosong and Zhang, Xian},
  booktitle={proceedings of the 21st international conference on computational linguistics and 44th annual meeting of the association for computational linguistics},
  pages={545--552},
  year={2006}
}
""",
        descriptive_stats={"n_samples": {"test": 100000}, "avg_character_length": None},
    )
