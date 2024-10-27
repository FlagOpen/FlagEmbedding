import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-4")
    parser.add_argument("--pred_path", default="output_dir/qwen/pred_summary_all.json", help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default="output_dir/qwen_subplot_all", help="The path to save annotation json files.")
    parser.add_argument("--output_json", default="output_dir/qwen_subplot_all_results.json", help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args



def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-4
    """
    # q_s_dict = get_scoring_points()
    for file in tqdm(caption_files):
        print("#############",file)
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        question = question.replace('\n','')
        answer = qa_set['a']
        pred = qa_set['pred']
        # scoring_points = q_s_dict[question]
        try:
            # Compute the correctness score
            completion = openai.ChatCompletion.create(
                temperature=0,
                model="gpt-4-turbo",
                messages = [
                    {
                        "role": "system",
                        "content": 
                        """
                            ##TASK DESCRIPTION: 
                            You are required to evaluate the performance of the respondent in the video summarization task based on the standard answer and the respondent's answer. You should provide two scores. The first is the COMPLETENESS score, which should range from 1 to 5. The second is the RELIABILITY score, which should also range from 1 to 5. Below are the criteria for each scoring category:
                            ##COMPLETENESS Scoring Criteria:
                            The completeness score focuses on whether the summary covers all key points and main information from the video. 
                            Score 1: The summary hardly covers any of the main content or key points of the video.
                            Score 2: The summary covers some of the main content and key points but misses many.
                            Score 3: The summary covers most of the main content and key points.
                            Score 4: The summary is very comprehensive, covering most to nearly all of the main content and key points.
                            Score 5: The summary completely covers all the main content and key points of the video.
                            ##RELIABILITY Scoring Criteria:
                            The reliability score evaluates the correctness and clarity of the video summary. It checks for factual errors, misleading statements, and contradictions with the video content. If the respondent's answer includes details that are not present in the standard answer, as long as these details do not conflict with the correct answer and are reasonable, points should not be deducted.
                            Score 1: Contains multiple factual errors and contradictions; presentation is confusing.
                            Score 2: Includes several errors and some contradictions; needs clearer presentation.
                            Score 3: Generally accurate with minor errors; minimal contradictions; reasonably clear presentation.
                            Score 4: Very accurate with negligible inaccuracies; no contradictions; clear and fluent presentation.
                            Score 5: Completely accurate with no errors or contradictions; presentation is clear and easy to understand.
                            ----
                            ##INSTRUCTION:
                            1. Evaluate COMPLETENESS: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence. 
                            2. Evaluate RELIABILITY: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence. 
                            3. Output Scores in JSON Format: Present the scores in JSON format as follows:
                            {'score_completeness': score_comp, 'score_reliability': score_reli, 'total_score': score_comp + score_reli}
                        """
                    },
                    {
                        "role": "user",
                        "content": f"""
                            Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
                            Standard Answer: {answer}
                            Respondent's Answer: {pred}
                        """
                    }
                ]

            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            # print("#############",response_message)
         
          
            save_dict={}
         
            # response_dict = ast.literal_eval(response_message)
            # qa_set["scoring_points"] = scoring_points
            save_dict["explain"] = response_message
            result_qa_pair = [save_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)
            
    

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    pred_contents = json.load(file)

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_name']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['video_name'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir

    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    


    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['video_name']
        question = sample['Q']
        answer = sample['A']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    # Set the OpenAI API key.
    openai.api_key = args.api_key
    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool(processes=1) as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")


    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")



if __name__ == "__main__":
    main()

