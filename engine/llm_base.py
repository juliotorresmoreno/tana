
import time
from decouple import config
from langchain.chains import RetrievalQA
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.base import Callbacks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loaders.elasticsearch import ElasticSearchLoader
from langchain.prompts import PromptTemplate


OLLAMA_TEXT_GENERATION_MODEL = config('OLLAMA_TEXT_GENERATION_MODEL')
template = PromptTemplate.from_template("[INST]<<SYS>>Soy una inteligencia artificial con intereses sociales, quiero ser amiga de todos.<</SYS>>\n{question}[/INST]")

class LLMBase:
    knowledge: ElasticSearchLoader

    def __init__(self) -> None:
        self.knowledge = ElasticSearchLoader()

    def get_context(self, index_name: str, question: str):
        if not index_name:
            return None

        data = self.knowledge.search(index_name=index_name, question=question)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)

        return Chroma.from_documents(
            documents=all_splits, embedding=GPT4AllEmbeddings())

    def invoke(self, question: str, index_name: str, callbacks: Callbacks):
        start_time = time.time()
        base_context = self.get_context(
            index_name=index_name, question=question)
        if base_context:
            chain_args = {
                "llm": self.provider,
                "chain_type_kwargs": {"prompt": template},
                'retriever': base_context.as_retriever()
            }
            qa_chain = RetrievalQA.from_chain_type(**chain_args)
            result = qa_chain({ "query": question }, callbacks=callbacks)['result']
        else:
            chain_args = {
                "llm": self.provider,
                "chain_type_kwargs": {"prompt": template}
            }
            result = self.provider(question, callbacks=callbacks)

        execution_time = time.time() - start_time

        return result, execution_time
