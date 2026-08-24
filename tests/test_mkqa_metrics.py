from FlagEmbedding.evaluation.mkqa.utils.compute_metrics import evaluate_qa_recall


def test_evaluate_qa_recall_supports_variable_retrieval_depths():
    contexts = [
        ["no answer here"],
        ["still no answer", "the target answer is here"],
    ]
    answers = [["target answer"], ["target answer"]]

    recall = evaluate_qa_recall(contexts, answers, k_values=[1, 2])

    assert recall == [0.0, 0.5]
