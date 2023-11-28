from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import FAISS

def compression(documents, question: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    retriever = FAISS.from_documents(texts, GPT4AllEmbeddings()).as_retriever()

    docs = retriever.get_relevant_documents(question)
    return docs
