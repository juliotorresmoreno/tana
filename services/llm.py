import threading
from queue import Queue
from engine.mmlu import engine
from callbacks.stream import StreamingStdOutCallbackHandlerYield, generate
from models.bot import Bot
from services.history import History

class LLM:
    def invoke(self,
        prompt: str,
        index_name: str, 
        bot: Bot,
        chat_history: History
    ):
        q = Queue()

        def ask_question(callback_fn: StreamingStdOutCallbackHandlerYield):
            return engine.invoke(prompt, index_name, [callback_fn])

        callback_fn = StreamingStdOutCallbackHandlerYield(q)
        threading.Thread(target=ask_question, args=(callback_fn,)).start()

        return generate(q)
