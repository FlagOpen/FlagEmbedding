# Overview
Here is the official implementation of the training data generation process for ***"Extending Llama-3's Context Ten-Fold Overnight"***. We utilized OpenAI's API to generate two main categories of data: *Book* and *Paper*. Additionally, we have further subdivided them into *"One-detail QA"*, *"Multi-detail QA"*, *"Biography Summary"*. You can refer to the code here based on your needs.

# File Structure
```bash
data_pipeline/
├── data
├── raw_data
├── prepare_bio_book.ipynb
├── prepare_multi_details_book.ipynb
├── prepare_multi_details_paper_long.ipynb
├── prepare_one_detail_book.ipynb
├── prepare_one_detail_paper_long.ipynb
├── _openai.py
└── README.md
```

# requirements
```
datasets==2.20.0
langdetect==1.0.9
semantic-text-splitter==0.13.3
tiktoken==0.7.0
```

# Usage
1. Firstly, download the raw dataset to `raw_data`. 

2. Secondly, run each notebook to process raw dataset. Each notebook will create a temporary request file in `data`, which is used for `_openai.py`.

3. Finally, put your own API key in `_openai.py`, then run following command, the result will be placed in `data` as `{file_name}.result.jsonl`: 
    ```shell
    python _openai.py --request ${file in `data`} 
    ```


# Note
- To prevent overlap of training data, we recommend that you execute the notebook in the following order: `prepare_one_detail_book` > `prepare_bio_book` > `prepare_multi_details_book` > `prepare_one_detail_paper_long` > `prepare_multi_details_paper_long`

