from FlagEmbedding import FlagModel
from CsvToList import read_csv_to_list
import numpy as np
import json

generated_CWEs = ["sample data-1", "foobar"]
actual_CWEs = read_csv_to_list("CWEdescriptions.csv", column_index=0)

model = FlagModel('BAAI/bge-large-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

embeddings_input = model.encode(generated_CWEs)
embeddings_actual = np.load('embeddings.npy')

# similarity matrix between CVEs and CWEs (CVEs x CWEs) where each element is the cosine similarity between the corresponding CVE and CWE
simularity = embeddings_input @ embeddings_actual.T

# Find the top 3 most similar CWEs for each CVE
top_cwes_indicies = np.argsort(simularity, axis=1)[:, -3:][:, ::-1]
top_cwes = []

for i, indicies in enumerate(top_cwes_indicies):
    similar_cwes = [actual_CWEs[index] for index in indicies]
    top_cwes.append(similar_cwes)

# we can reference the similarity matrix to get the similarity between a specific CVE and CWE
data = {
    "generated_CWEs": generated_CWEs,
    "embeddings_input": embeddings_input.tolist(),
    "embeddings_actual": embeddings_actual.tolist(),
    "similarity": simularity.tolist(),
    "top_CWEs": top_cwes
}

with open("similarity.json", "w") as file:
    json.dump(data, file, indent=4)