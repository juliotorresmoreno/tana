import threading
from queue import Queue
from callbacks.stream import StreamingStdOutCallbackHandlerYield, generate
from models.bot import Bot
from services.history import History
from decouple import config
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from services.retriever import Retriever

OLLAMA_TEXT_GENERATION_MODEL = config('OLLAMA_TEXT_GENERATION_MODEL')

def trim_sys_info(input_string: str):
    start_delimiters = ["<<SYS>> content='", "<<SYS>>", "content='"]
    end_delimiters = ["' <</SYS>>", "<</SYS>>", "'"]
    result = input_string

    for start_delimiter in start_delimiters:
        if input_string.startswith(start_delimiter):
            input_string[len(start_delimiter):]
            break

    for end_delimiter in end_delimiters:
        if input_string.endswith(end_delimiter):
            input_string[:-len(start_delimiter)]
            break

    return result

class LLM:    
    retriever = Retriever()
    
    def invoke(self,
        prompt: str,
        loaders: list, 
        bot: Bot,
        chat_history: History
    ):                 
        q = Queue()
        callback_fn = StreamingStdOutCallbackHandlerYield(q)
                
        def ask_question(callback_fn: StreamingStdOutCallbackHandlerYield):
            chat_model = ChatOllama(
                model="llama2",
                format="json",
                callbacks=[callback_fn]
            )
            chat_history.add_title(bot['description']) 
            messages = [SystemMessage(content=bot["description"])]
                
            if loaders != None and len(loaders) > 0:
                relevant = self.retriever.search(
                    loaders=loaders,
                    question=prompt,
                    history=chat_history.get_documents(query=prompt, format='documents')
                )
                if len(relevant) > 0:
                    messages.append(HumanMessage(content=relevant))

            messages.append(HumanMessage(content=prompt))
            response = chat_model(messages)
            response = trim_sys_info(response.content)
            
            chat_history.add_human_message(prompt)
            chat_history.add_system_message(str(response))

        
        threading.Thread(target=ask_question, args=(callback_fn,)).start()
        
        return generate(q)
    