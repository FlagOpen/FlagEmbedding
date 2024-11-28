import platform
import os
import subprocess
    
import numpy as np
import pandas as pd

def run_cmd(cmd):
    """Execute a shell command and return its output."""
    try:
        return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    except:
        return None

def get_env_info():
    """
    Detect the current running environment.
    Returns: str, str - One of "Kaggle", "Mac", "AutoDL", "Linux", or "Unknown" and project root
    """
    # Check Kaggle first
    hostname = run_cmd('hostname')  # machine hostname
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        return "Kaggle", None
    
    # Check if Mac or Linux
    system = platform.system()
    if system == "Darwin":
        return "Mac", '/Users/runshengliu/github/kaggle-eedi-math'
    elif system == "Linux":
        if hostname and 'autodl' in hostname:
            return "AutoDL", '/root/autodl-tmp/github/kaggle-eedi-math'
        else:
            return "Linux", '/home/ubuntu/kaggle-eedi-math'
    else:
        return "Unknown", None

def prepare_val_data_v1(env, PROJECT_ROOT=None):
    np.random.seed(42)
    if env == "Kaggle":
        raw_data = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
    else:
        raw_data = pd.read_csv(f"{PROJECT_ROOT}/data/train.csv")
    print(f"Original raw_data shape: {raw_data.shape}")
    
    # randomly choose 33.3% QuestionId from train_data as validation set, and the rest as training set
    # Sort QuestionIds to ensure consistent ordering before selection
    question_ids = sorted(raw_data["QuestionId"].unique())
    val_question_ids = np.random.choice(question_ids, int(len(raw_data) * 0.333), replace=False)
    
    val_data = raw_data[raw_data["QuestionId"].isin(val_question_ids)]
    train_data = raw_data[~raw_data["QuestionId"].isin(val_question_ids)]
    print(f"val_data shape: {val_data.shape}")
    print(f"train_data shape: {train_data.shape}")
    return train_data, val_data

def prepare_val_data_v2(env, PROJECT_ROOT=None):
    np.random.seed(42)
    if env == "Kaggle":
        raw_data = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
    else:
        raw_data = pd.read_csv(f"{PROJECT_ROOT}/data/train.csv")
    print(f"Original raw_data shape: {raw_data.shape}")
    
    # randomly choose 33.3% QuestionId from train_data as validation set (all the rows contains no NaN MisconceptionId), and the rest as training set
    qualified_question_ids = []
    for id, row in raw_data.iterrows():
        correct_answer = row['CorrectAnswer']
        wrong_answers = [f'Misconception{answer}Id' for answer in ['A', 'B', 'C', 'D'] if answer != correct_answer]
        if all(row[wrong_answers].notna()):
            qualified_question_ids.append(row['QuestionId'])
    print(f"qualified_question_ids shape: {len(qualified_question_ids)}")
    qualified_question_ids = sorted(qualified_question_ids)
    
    val_question_ids = np.random.choice(qualified_question_ids, int(len(raw_data) * 0.333), replace=False)
    val_data = raw_data[raw_data["QuestionId"].isin(val_question_ids)]
    train_data = raw_data[~raw_data["QuestionId"].isin(val_question_ids)]
    print(f"val_data shape: {val_data.shape}")
    print(f"train_data shape: {train_data.shape}")
    return train_data, val_data


if __name__ == "__main__":
    env, PROJECT_ROOT = get_env_info()
    print(f"Running on {env}")
    print(f"Project root: {PROJECT_ROOT}")

    train_data_v1, val_data_v1 = prepare_val_data_v1(env, PROJECT_ROOT)
    train_data_v2, val_data_v2 = prepare_val_data_v2(env, PROJECT_ROOT)

    train_data_v1.to_csv(f"/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/eval_data/raw_data/validation_v1/train.csv", index=False)
    val_data_v1.to_csv(f"/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/eval_data/raw_data/validation_v1/val.csv", index=False)
    train_data_v2.to_csv(f"/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/eval_data/raw_data/validation_v2/train.csv", index=False)
    val_data_v2.to_csv(f"/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/eval_data/raw_data/validation_v2/val.csv", index=False)
