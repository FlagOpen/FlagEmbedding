import json
import re
import os


path="/output_dir/videollama7b_summary_all"
def extract_scores(text):
    # Define the keys to locate in the text
    keys = ["score_completeness", "score_reliability"]
    scores = []

    for key in keys:
        # Find the index where each key starts
        start_index = text.find(key)
        if start_index == -1:
            continue  # Skip if key is not found

        # Find the start of the number which is after the colon and space
        start_number_index = text.find(":", start_index) + 2
        end_number_index = text.find(",", start_number_index)  # Assuming the number ends before a comma

        # Extract and convert the number to float
        score = float(text[start_number_index:end_number_index])
        scores.append(score)

    return scores



accu=0
rele=0
total=0
file_list=os.listdir(path)



for i in file_list:
    file_path=os.path.join(path,i)
    with open(file_path,"r") as f:
        data=json.load(f)

    # print(file_path)
    text=data[0]["explain"]
    # print(text)
    scores=extract_scores(text)
    # print("score",scores)

    try:
        accu += scores[0]
        rele += scores[1]
    except:
        accu +=0
        rele+=0
  

accu = accu/ len(file_list)
rele = rele/ len(file_list)
total= (accu + rele ) 

print(accu,rele,total)