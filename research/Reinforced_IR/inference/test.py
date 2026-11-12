from ir_model import Reinforced_IR_Model

api_key="sk-gzAdunPMOSEDdotUkMgwnHKN5eP4a2vZx8GKBeN1hHH017z0"
base_url="https://api.xiaoai.plus/v1"

model = Reinforced_IR_Model(
	model_name_or_path='/share/chaofan/models/bge-large-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    use_fp16=True,
    devices=['cuda:0'],
    # generator_model_name_or_path='gpt-4o-mini',
    api_key=api_key,
    base_url=base_url,
    temperature=0,
    model_type='gpt' # gpt, llm, llm_instruct
)

queries = ["how much protein should a female eat", "summit define"]

documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]

task_instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
answer_type = 'passage'

embeddings_1 = model.encode_queries(
    task_instruction=task_instruction,
    answer_type=answer_type,
    queries=queries
)
embeddings_2 = model.encode_corpus(documents)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)