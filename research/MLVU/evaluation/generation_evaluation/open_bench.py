import torch
from torchvision import transforms
import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

class MLVU(Dataset):
    def __init__(self, data_dir, data_list):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'data': data
                })
        
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    

    def qa_template(self, data):
        question = f"{data['question']}"
        answer = data['answer']
        return question, answer


    def __getitem__(self, idx):
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }



def main():

    data_list = {
    "subPlot": ("8_sub_scene.json", f"MLVU_all/video/subPlot", "video", False),
    "summary": ("9_summary.json", f"MLVU_all/video/summary", "video", False)
    }
   

    data_dir = f"MLVU_all/json"
   

    dataset = MLVU(data_dir, data_list)

    '''
    load your model
    '''

    res_list_subplot = []
    res_list_summary = []
    for example in tqdm(dataset):
        video_path=example["video"]
        quesiotn=example["question"]

        '''
        modify the conv_templates like the following tempates
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        inp = [DEFAULT_VIDEO_TOKEN] '\n' + question  # noted different models take different concatenation ways
        conv.system="Carefully watch this video and pay attention to every detail. Based on your observations, answer the given questions."
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt=get_prompt(conv)
        '''

        '''
        run the inference code of MLLMs
        pred=run(video_path,conv_mode,prompt,...)
        '''
    
        gt = example['answer']

        if task_type=="subPlot":
            result={}
            result["video_name"]=example['video_path'].split("/")[-1]
            result['Q']=example['question']
            result['A']=gt
            result['pred']=pred
            res_list_subplot.append(result)
    
        if task_type=="summary":
            result={}
            result["video_name"]=example['video_path'].split("/")[-1]
            result['Q']=example['question']
            result['A']=gt
            result['pred']=pred
            res_list_summary.append(result)



    with open(f"subplot_all.json", "w") as f:
        json.dump(res_list_subplot, f)

    with open(f"summary_all.json", "w") as f:
        json.dump(res_list_summary, f)

      

if __name__ == '__main__':
    main()
