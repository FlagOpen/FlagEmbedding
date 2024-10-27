from mteb import AbsTaskSTS


class ATEC(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "ATEC",
            "hf_hub_name": "C-MTEB/ATEC",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
        }



class BQ(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "BQ",
            "hf_hub_name": "C-MTEB/BQ",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
        }


class LCQMC(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "LCQMC",
            "hf_hub_name": "C-MTEB/LCQMC",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
        }



class PAWSX(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "PAWSX",
            "hf_hub_name": "C-MTEB/PAWSX",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
        }


class STSB(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "STSB",
            "hf_hub_name": "C-MTEB/STSB",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
        }


class AFQMC(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "AFQMC",
            "hf_hub_name": "C-MTEB/AFQMC",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
        }



class QBQTC(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "QBQTC",
            "hf_hub_name": "C-MTEB/QBQTC",
            "reference": "https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
        }
