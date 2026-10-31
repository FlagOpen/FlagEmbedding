"""
Test loading MSMARCO qrels from remote without a local ``dataset_dir``.

The download is mocked with a tiny TREC-style qrels file, so no network access is needed.
"""

import pytest

from FlagEmbedding.evaluation.msmarco import MSMARCOEvalDataLoader


@pytest.mark.parametrize("dataset_name", ["passage", "document"])
@pytest.mark.parametrize("split", ["dl19", "dl20"])
def test_load_remote_qrels_without_save_dir(tmp_path, monkeypatch, dataset_name, split):
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("1 Q0 doc-1 2\n1 Q0 doc-2 0\n2 Q0 doc-3 1\n", encoding="utf-8")

    data_loader = MSMARCOEvalDataLoader(eval_name="msmarco", cache_dir=str(tmp_path))
    monkeypatch.setattr(data_loader, "_download_file", lambda *args, **kwargs: str(qrels_file))

    qrels = data_loader.load_qrels(dataset_name=dataset_name, split=split)

    assert dict(qrels) == {"1": {"doc-1": 2, "doc-2": 0}, "2": {"doc-3": 1}}
