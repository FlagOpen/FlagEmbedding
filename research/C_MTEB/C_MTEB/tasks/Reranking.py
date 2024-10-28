from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class T2Reranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="T2Reranking",
        description="T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
        reference="https://arxiv.org/abs/2304.03679",
        dataset={
            "path": "C-MTEB/T2Reranking",
            "revision": "76631901a18387f85eaa53e5450019b87ad58ef9",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="map",
        date=None,
        form=None,
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
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )


class MMarcoReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="MMarcoReranking",
        description="mMARCO is a multilingual version of the MS MARCO passage ranking dataset",
        reference="https://github.com/unicamp-dl/mMARCO",
        dataset={
            "path": "C-MTEB/Mmarco-reranking",
            "revision": "8e0c766dbe9e16e1d221116a3f36795fbade07f6",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="map",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{bonifacio2021mmarco,
      title={mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset}, 
      author={Luiz Henrique Bonifacio and Vitor Jeronymo and Hugo Queiroz Abonizio and Israel Campiotti and Marzieh Fadaee and  and Roberto Lotufo and Rodrigo Nogueira},
      year={2021},
      eprint={2108.13897},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )


class CMedQAv1(AbsTaskReranking):
    metadata = TaskMetadata(
        name="CMedQAv1-reranking",
        description="Chinese community medical question answering",
        reference="https://github.com/zhangsheng93/cMedQA",
        dataset={
            "path": "C-MTEB/CMedQAv1-reranking",
            "revision": "8d7f1e942507dac42dc58017c1a001c3717da7df",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="map",
        date=("2017-01-01", "2017-07-26"),
        domains=["Medical", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{zhang2017chinese,
  title={Chinese Medical Question Answer Matching Using End-to-End Character-Level Multi-Scale CNNs},
  author={Zhang, Sheng and Zhang, Xin and Wang, Hui and Cheng, Jiajun and Li, Pei and Ding, Zhaoyun},
  journal={Applied Sciences},
  volume={7},
  number={8},
  pages={767},
  year={2017},
  publisher={Multidisciplinary Digital Publishing Institute}
}""",
        descriptive_stats={
            "n_samples": {"test": 2000},
            "avg_character_length": {"test": 165},
        },
    )


class CMedQAv2(AbsTaskReranking):
    metadata = TaskMetadata(
        name="CMedQAv2-reranking",
        description="Chinese community medical question answering",
        reference="https://github.com/zhangsheng93/cMedQA2",
        dataset={
            "path": "C-MTEB/CMedQAv2-reranking",
            "revision": "23d186750531a14a0357ca22cd92d712fd512ea0",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="map",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@ARTICLE{8548603, 
author={S. Zhang and X. Zhang and H. Wang and L. Guo and S. Liu}, 
journal={IEEE Access}, 
title={Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection}, 
year={2018}, 
volume={6}, 
number={}, 
pages={74061-74071}, 
keywords={Biomedical imaging;Data mining;Semantics;Medical services;Feature extraction;Knowledge discovery;Medical question answering;interactive attention;deep learning;deep neural networks}, 
doi={10.1109/ACCESS.2018.2883637}, 
ISSN={2169-3536}, 
month={},}""",
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )
