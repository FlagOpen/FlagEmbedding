import platform
import os
import subprocess
import argparse
import json

# Add str2bool helper function
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# Add argument parser
parser = argparse.ArgumentParser(description='Eval data preprocessing script for EEDI Competition')
parser.add_argument('--filter_na_misconception', type=str2bool, nargs='?', const=True, default=False,
                   help='Whether to filter out rows with NA in MisconceptionId (true/false)')
parser.add_argument('--with_instruction', type=str2bool, nargs='?', const=True, default=False,
                   help='Whether to add instruction to the query (true/false)')
parser.add_argument('--query_text_version', type=str, choices=['v1', 'v2', 'v3'], default='v1',
                   help='Query text version')
args = parser.parse_args()

# Add this after parsing arguments
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print() 

import pandas as pd

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_data(train_data, 
                    misconception_mapping, 
                    query_text_version, 
                    with_instruction=True, 
                    with_misconception=True, 
                    filter_na_misconception=True):

    # 1. Melt answer columns and create base dataframe
    answer_cols = ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']
    answer_values = ['A', 'B', 'C', 'D']

    # Melt the answer columns
    melted_answers = pd.melt(
        train_data,
        id_vars=['QuestionId', 'QuestionText', 'ConstructId', 'ConstructName', 
                'SubjectId', 'SubjectName', 'CorrectAnswer'],
        value_vars=answer_cols,
        var_name='AnswerColumn',
        value_name='WrongAnswerText'
    )
    # Add WrongAnswer column based on AnswerColumn
    melted_answers['WrongAnswer'] = melted_answers['AnswerColumn'].map(
        dict(zip(answer_cols, answer_values))
    )


    # 2. Add MisconceptionId and MisconceptionName if with_misconception = True
    if with_misconception:
        misconception_cols = [f'Misconception{x}Id' for x in ['A', 'B', 'C', 'D']]  # Fixed column names
        melted_misconceptions = pd.melt(
            train_data,
            id_vars=['QuestionId', 'CorrectAnswer'],
            value_vars=misconception_cols,
            var_name='MisconceptionColumn',
            value_name='MisconceptionId'
        )
        melted_misconceptions['WrongAnswer'] = melted_misconceptions['MisconceptionColumn'].str[-3]
        
        df = melted_answers.merge(
            melted_misconceptions[['QuestionId', 'WrongAnswer', 'MisconceptionId']], 
            on=['QuestionId', 'WrongAnswer'], 
            how='left'
        )

        df = df.merge(
            misconception_mapping[['MisconceptionId', 'MisconceptionName']], 
            on='MisconceptionId', 
            how='left'
        )
    else:
        df = melted_answers

    # Create CorrectAnswerText column
    correct_answers = df[['QuestionId', 'WrongAnswer', 'WrongAnswerText']].copy()
    correct_answers = correct_answers[
        correct_answers['WrongAnswer'] == correct_answers['QuestionId'].map(
            train_data.set_index('QuestionId')['CorrectAnswer']
        )
    ]
    correct_answers = correct_answers.rename(
        columns={'WrongAnswerText': 'CorrectAnswerText'}
    )[['QuestionId', 'CorrectAnswerText']]
    # Merge correct answer text
    df = df.merge(correct_answers, on='QuestionId', how='left')
    # Filter out the correct answer
    df = df[df['WrongAnswer'] != df['CorrectAnswer']]
    # Create QuestionId_Answer column
    df['QuestionId_Answer'] = df['QuestionId'].astype(str) + '_' + df['WrongAnswer']
    if with_misconception:
        final_columns = ['QuestionId_Answer', 'QuestionId', 'QuestionText', 'ConstructId',
            'ConstructName', 'SubjectId', 'SubjectName', 'CorrectAnswer', 'CorrectAnswerText',
            'WrongAnswerText', 'WrongAnswer', 'MisconceptionId', 'MisconceptionName']
    else:
        final_columns = ['QuestionId_Answer', 'QuestionId', 'QuestionText', 'ConstructId',
            'ConstructName', 'SubjectId', 'SubjectName', 'CorrectAnswer', 'CorrectAnswerText',
            'WrongAnswerText', 'WrongAnswer']
    df = df[final_columns]
    
    if query_text_version == "v1":
        df["query_text"] = df["ConstructName"] + " " + df["QuestionText"] + " " + df["WrongAnswerText"]
        df["query_text"] = df["query_text"].apply(preprocess_text)
    else:
        raise ValueError(f"Invalid query_text_version: {query_text_version}")
    
    if with_instruction:
        task_description = 'Given a math question and an incorrect answer, please retrieve the most accurate reason for the misconception leading to the incorrect answer.'
        df['query_text'] = df.apply(lambda row: f"Instruction:{task_description}\nQuery:{row['query_text']}", axis=1)

    # filter out rows with NA in MisconceptionId
    if with_misconception and filter_na_misconception:
        df = df[df['MisconceptionId'].notna()]
    
    df = df.sort_values(['QuestionId', 'QuestionId_Answer']).reset_index(drop=True)
    df['order_index'] = df['QuestionId_Answer']
    
    return df

if __name__ == "__main__":
    RAW_DATA_DIR = "/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/eval_data/raw_data"
    EVAL_DATA_DIR = "/root/autodl-tmp/github/FlagEmbedding/examples/finetune/embedder/eval_data"
    misconception_mapping = pd.read_csv(f"{RAW_DATA_DIR}/misconception_mapping.csv")
    misconception_mapping['MisconceptionName'] = misconception_mapping['MisconceptionName'].apply(preprocess_text)
    corpus = misconception_mapping['MisconceptionName'].values.tolist()
        
    
    with open(f"{EVAL_DATA_DIR}/corpus.jsonl", "w", encoding="utf-8") as f:
        for sentence in corpus:
            json_line = {"text": sentence}  # 将每一行封装为字典
            f.write(json.dumps(json_line) + "\n")
    
    for version in ["v1", "v2"]:
        val_data = pd.read_csv(f"{RAW_DATA_DIR}/validation_{version}/val.csv")
        val_preprocessed = preprocess_data(val_data, misconception_mapping, 
                                    query_text_version=args.query_text_version,
                                    with_instruction=args.with_instruction, 
                                    with_misconception=True, 
                                    filter_na_misconception=args.filter_na_misconception)
        
        selected_columns = ["query_text", "MisconceptionId"]
        df_selected = val_preprocessed[selected_columns]

        # 将 JSON 数据写入文件
        with open(f"{EVAL_DATA_DIR}/queries_{version}.jsonl", "w") as f:
            for _, row in df_selected.iterrows():
                json_line = {"query": row['query_text'], "correct_id": row['MisconceptionId']}
                f.write(json.dumps(json_line) + "\n")