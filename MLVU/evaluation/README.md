## Evaluation for MLVU

We provide detailed evaluation methods for MLVU, including Multiple-choice tasks and generation tasks.

### Benchmark MLVU on your Model
Firstly, If you want to benchmark MLVU in your models, you can refer to our template test code as follows:
#### Multiple-Choice testing
```
python multiple_choice_evaluation/choice_bench.py 
```
You must load your model into this template and evaluate the multiple-choice performance online. 
#### Generation testing
- Step 1 Get the inference results of Sub-Scene Captioning and Video Summary.
```
python generation_evaluation/open_bench.py 
```
- Step 2 Run the evaluation for the generation tasks.
For Sub-Scene Captioning, modify your pred_path (by step 1) and output_dir then run
```
python evaluate_ssc.py --pred_path /your_path/subplot_all.json --output_dir /eval_subplot  --output_json /eval_subplot.json
python calculate.py --path /eval_subplot
```
For Video Summarization, modify your pred_path (by step 1) and output_dir then run
```
python evaluate_summary.py --pred_path /your_path/summary_all.json --output_dir /eval_summary  --output_json /eval_summary.json
```
Then run, and you need to modify the path in it to your output_dir
```
python calculate_sum.py --path /eval_summary
```

### Benchmark MLVU on existing models
(Take VideoChat2 as an example:)
- step 1: Download original models as well as weights from [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)
- step 2: Put choice_bench.py and open_bench.py into the folder as the same as demo.py  
- step 3: modify your path of the MLVU in choice_bench.py and open_bench.py
- step 4: run the inference and online evaluation for Multiple-choice tasks.
- step 5: run the inference and evaluation for the generation tasks. 
