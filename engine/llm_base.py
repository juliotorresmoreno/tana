
from langchain.chains import RetrievalQA
from langchain.llms import base
from langchain.callbacks.base import Callbacks
from langchain import hub
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loaders.elasticsearch import ElasticSearchLoader
from langchain.llms.llamacpp import LlamaCpp
import time
from decouple import config


OLLAMA_TEXT_GENERATION_MODEL = config('OLLAMA_TEXT_GENERATION_MODEL')
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
QA_CHAIN_PROMPT.messages[0].prompt.template = "[INST]<<SYS>>Soy una inteligencia artificial con intereses sociales, quiero ser amiga de todos.<</SYS>>\n{question}[/INST]"

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

    def invoke(self, question: str, llm: base, index_name: str = None, callbacks: Callbacks = None):
        if len(question) < 10:
            return 'Please generate a longer question so I can respond based on the context.', 0

        start_time = time.time()
        base_context = self.get_context(
            index_name=index_name, question=question)
        if base_context:
            chain_args = {
                "llm": llm,
                "chain_type_kwargs": {"prompt": QA_CHAIN_PROMPT},
                'retriever': base_context.as_retriever()
            }
            qa_chain = RetrievalQA.from_chain_type(**chain_args)
            result = qa_chain({"query": question},
                              callbacks=callbacks)['result']
            execution_time = time.time() - start_time
            return result, execution_time

        result = llm.invoke(question)
        execution_time = time.time() - start_time

        return result, execution_time
