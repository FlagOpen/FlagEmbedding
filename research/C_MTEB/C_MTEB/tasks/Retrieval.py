from __future__ import annotations

from collections import defaultdict

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


def load_retrieval_data(dataset_path, dataset_revision, qrel_revision, eval_splits):
    eval_split = eval_splits[0]
    dataset = load_dataset(dataset_path, revision=dataset_revision)
    qrels = load_dataset(dataset_path + "-qrels", revision=qrel_revision)[eval_split]

    corpus = {e["id"]: {"text": e["text"]} for e in dataset["corpus"]}
    queries = {e["id"]: e["text"] for e in dataset["queries"]}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e["qid"]][e["pid"]] = e["score"]

    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    return corpus, queries, relevant_docs


class T2Retrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="T2Retrieval",
        description="T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
        reference="https://arxiv.org/abs/2304.03679",
        dataset={
            "path": "C-MTEB/T2Retrieval",
            "revision": "8731a845f1bf500a4f111cf1070785c793d10e64",
            "qrel_revision": "1c83b8d1544e529875e3f6930f3a1fcf749a8e97",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{xie2023t2ranking,
      title={T2Ranking: A large-scale Chinese Benchmark for Passage Ranking}, 
      author={Xiaohui Xie and Qian Dong and Bingning Wang and Feiyang Lv and Ting Yao and Weinan Gan and Zhijing Wu and Xiangsheng Li and Haitao Li and Yiqun Liu and Jin Ma},
      year={2023},
      eprint={2304.03679},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 874.1184182791619,
                    "average_query_length": 10.938847974750132,
                    "num_documents": 118605,
                    "num_queries": 22812,
                    "average_relevant_docs_per_query": 5.213571804313519,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class MMarcoRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MMarcoRetrieval",
        description="MMarcoRetrieval",
        reference="https://arxiv.org/abs/2309.07597",
        dataset={
            "path": "C-MTEB/MMarcoRetrieval",
            "revision": "539bbde593d947e2a124ba72651aafc09eb33fc2",
            "qrel_revision": "bae08bb7bddbedb96c7e7db52018a55167b67f89",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{xiao2024cpack,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
      year={2024},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 114.41787048392986,
                    "average_query_length": 10.51131805157593,
                    "num_documents": 106813,
                    "num_queries": 6980,
                    "average_relevant_docs_per_query": 1.0654727793696275,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class DuRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DuRetrieval",
        description="A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        dataset={
            "path": "C-MTEB/DuRetrieval",
            "revision": "a1a333e290fe30b10f3f56498e3a0d911a693ced",
            "qrel_revision": "497b7bd1bbb25cb3757ff34d95a8be50a3de2279",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{qiu2022dureaderretrieval,
      title={DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine}, 
      author={Yifu Qiu and Hongyu Li and Yingqi Qu and Ying Chen and Qiaoqiao She and Jing Liu and Hua Wu and Haifeng Wang},
      year={2022},
      eprint={2203.10232},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 331.3219967800322,
                    "average_query_length": 9.289,
                    "num_documents": 100001,
                    "num_queries": 2000,
                    "average_relevant_docs_per_query": 4.9195,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CovidRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CovidRetrieval",
        description="COVID-19 news articles",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "C-MTEB/CovidRetrieval",
            "revision": "1271c7809071a13532e05f25fb53511ffce77117",
            "qrel_revision": "a9f41b7cdf24785531d12417ce0d1157ed4b39ca",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 332.4152658473415,
                    "average_query_length": 25.9304531085353,
                    "num_documents": 100001,
                    "num_queries": 949,
                    "average_relevant_docs_per_query": 1.0105374077976819,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CmedqaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CmedqaRetrieval",
        description="Online medical consultation text. Used the CMedQAv2 as its underlying dataset.",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        dataset={
            "path": "C-MTEB/CmedqaRetrieval",
            "revision": "cd540c506dae1cf9e9a59c3e06f42030d54e7301",
            "qrel_revision": "279d737f36c731c8ff6e2b055f31fe02216fa23d",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 307.7710222897771,
                    "average_query_length": 48.470367591897976,
                    "num_documents": 100001,
                    "num_queries": 3999,
                    "average_relevant_docs_per_query": 1.86271567891973,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class EcomRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="EcomRetrieval",
        description="EcomRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "C-MTEB/EcomRetrieval",
            "revision": "687de13dc7294d6fd9be10c6945f9e8fec8166b9",
            "qrel_revision": "39c90699b034ec22ac45b3abf5b0bbb5ffd421f9",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 32.98041664189015,
                    "average_query_length": 6.798,
                    "num_documents": 100902,
                    "num_queries": 1000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class MedicalRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MedicalRetrieval",
        description="MedicalRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "C-MTEB/MedicalRetrieval",
            "revision": "2039188fb5800a9803ba5048df7b76e6fb151fc6",
            "qrel_revision": "37b8efec53c54c3d9c6af212f6710b62ccdf895c",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 122.04231725066585,
                    "average_query_length": 17.938,
                    "num_documents": 100999,
                    "num_queries": 1000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class VideoRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="VideoRetrieval",
        description="VideoRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "C-MTEB/VideoRetrieval",
            "revision": "58c2597a5943a2ba48f4668c3b90d796283c5639",
            "qrel_revision": "faa71382b6a29cf1778d1f436b963e75cb5b927c",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 31.048855642524522,
                    "average_query_length": 7.365,
                    "num_documents": 100930,
                    "num_queries": 1000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True
