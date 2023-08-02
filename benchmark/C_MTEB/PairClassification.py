from mteb import AbsTaskPairClassification


class Ocnli(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            'name': 'Ocnli',
            "hf_hub_name": "C-MTEB/OCNLI",
            'description': 'Original Chinese Natural Language Inference dataset',
            "reference": "https://arxiv.org/abs/2010.05444",
            'category': 's2s',
            'type': 'PairClassification',
            'eval_splits': ['validation'],
            'eval_langs': ['zh'],
            'main_score': 'ap',
        }


class Cmnli(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            'name': 'Cmnli',
            "hf_hub_name": "C-MTEB/CMNLI",
            'description': 'Chinese Multi-Genre NLI',
            "reference": "https://huggingface.co/datasets/clue/viewer/cmnli",
            'category': 's2s',
            'type': 'PairClassification',
            'eval_splits': ['validation'],
            'eval_langs': ['zh'],
            'main_score': 'ap',
        }
