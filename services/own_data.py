import threading
from queue import Queue
from callbacks.stream import StreamingStdOutCallbackHandlerYield, generate
from models.bot import Bot
from services.history import History
from decouple import config
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


OLLAMA_TEXT_GENERATION_MODEL = config('OLLAMA_TEXT_GENERATION_MODEL')

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

class LLM:
    provider: Ollama
    
    def __init__(self) -> None:
        self.provider = Ollama(
            model=OLLAMA_TEXT_GENERATION_MODEL,
            verbose=True,
            top_p=.9,
            top_k=40,
            temperature=.9,
        )
    
    def invoke(self,
        prompt: str,
        index_name: str, 
        bot: Bot,
        chat_history: History
    ):
        chat_history.add_title(bot['description'])        
        documents = chat_history.get_documents()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100)
        all_splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
        
        q = Queue()
        callback_fn = StreamingStdOutCallbackHandlerYield(q)
        
        def ask_question(callback_fn: StreamingStdOutCallbackHandlerYield):
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            qa = RetrievalQA.from_llm(self.provider, retriever=retriever)

            result = qa({"query": prompt }, callbacks=[callback_fn])
            print(result)
        
        threading.Thread(target=ask_question, args=(callback_fn,)).start()
        
        return generate(q)
    