
from typing import List
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from typing import Callable, TypedDict

class SecureLoader(TypedDict):
    condition: Callable[[str], bool]
    loader: BaseLoader

class Retriever:
    loaders = {
        "wiki": SecureLoader(loader=WikipediaLoader, condition=lambda n: len(n) > 20)
    }
    
    def search(self, loaders: List[str], question: str, history=List[Document]):
        documents = []
        
        documents.extend(history)
        
        if len(question.split()) > 5:
            for loader in loaders:
                if loader in self.loaders:
                    secureLoader = self.loaders[loader]
                    if secureLoader['condition'](question):
                        docs = secureLoader['loader'](query=question, load_max_docs=5).load()
                        documents.extend(docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        all_splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
        
        relevants: List[Document] = vectorstore.similarity_search(question)
        context = ''.join([ doc.page_content for doc in relevants])
        
        return context
