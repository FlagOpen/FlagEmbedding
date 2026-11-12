# encoding=gbk
import faiss
import numpy as np
import tiktoken
import torch
from datasets import load_from_disk
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModel


class LocalDatasetLoader:
    data_path: str = ""
    doc_emb: np.ndarray = None

    def __init__(self,
                 data_path,
                 embedding_path):
        dataset = load_from_disk(data_path)
        title = dataset['title']
        text = dataset['text']
        self.data = [title[i] + ' -- ' + text[i] for i in range(len(title))]
        self.doc_emb = np.load(embedding_path)


class QueryGenerator:
    def __init__(self):
        prompt_template = """请基于历史，对新问题进行修改，如果新问题与历史相关，你必须结合语境将代词替换为相应的指代内容，让它的提问更加明确；否则保留原始的新问题。只修改新问题，不作任何回答：\n历史：{history}\n新问题：{question}\n修改后的新问题："""
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        prompt = PromptTemplate(
            input_variables=["history", "question"],
            template=prompt_template,
        )

        self.llm_chain = LLMChain(llm=llm, prompt=prompt)

    def run(self, history, question):
        return self.llm_chain.predict(history=history, question=question).strip()


class AnswerGenerator:
    def __init__(self):
        prompt_template = """请基于参考，回答问题，不需要标注任何引用：\n历史：{history}\n问题：{question}\n参考：{references}\n答案："""
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        prompt = PromptTemplate(
            input_variables=["history", "question", "references"],
            template=prompt_template,
        )

        self.llm_chain = LLMChain(llm=llm, prompt=prompt)

    def run(self, history, question, references):
        return self.llm_chain.predict(history=history, question=question, references=references).strip()


class BMVectorIndex:
    def __init__(self,
                 model_path,
                 bm_index_path,
                 data_loader):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.bm_searcher = LuceneSearcher(bm_index_path)
        self.loader = data_loader

        if '-en' in model_path:
            raise NotImplementedError("only support chinese currently")

        if 'noinstruct' in model_path:
            self.instruction = None
        else:
            self.instruction = "为这个句子生成表示以用于检索相关文章："

    def search_for_doc(self, query: str, RANKING: int = 1000, TOP_N: int = 5):
        hits = self.bm_searcher.search(query, RANKING)
        ids = [int(e.docid) for e in hits]
        use_docs = self.loader.doc_emb[ids]

        if self.instruction is not None:
            query = self.instruction + query
        encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            query_emb = model_output.last_hidden_state[:, 0]
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=-1).detach().cpu().numpy()

        temp_index = faiss.IndexFlatIP(use_docs[0].shape[0])
        temp_index.add(use_docs)

        _, I = temp_index.search(query_emb, TOP_N)

        return "\n".join([self.loader.data[ids[I[0][i]]] for i in range(TOP_N)])


class Agent:
    def __init__(self, index):
        self.memory = ""
        self.index = index
        self.query_generator = QueryGenerator()
        self.answer_generator = AnswerGenerator()

    def empty_memory(self):
        self.memory = ""

    def update_memory(self, question, answer):
        self.memory += f"问：{question}\n答：{answer}\n"
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        while len(encoding.encode(self.memory)) > 3500:
            pos = self.memory[2:].rfind("问")
            self.memory = self.memory[pos:]

    def generate_query(self, question):
        if self.memory == "":
            return question
        else:
            return self.query_generator.run(self.memory, question)

    def generate_answer(self, query, references):
        return self.answer_generator.run(self.memory, query, references)

    def answer(self, question, RANKING=1000, TOP_N=5, verbose=True):
        query = self.generate_query(question)
        references = self.index.search_for_doc(query, RANKING=RANKING, TOP_N=TOP_N)
        answer = self.generate_answer(question, references)
        self.update_memory(question, answer)
        if verbose:
            print('\033[96m' + "查：" + query + '\033[0m')
            print('\033[96m' + "参：" + references + '\033[0m')
        print("答：" + answer)
