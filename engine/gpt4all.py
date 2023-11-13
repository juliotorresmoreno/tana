
from callbacks.realtime import StreamingWebsocketCallbackHandler
from langchain.callbacks.manager import CallbackManager
from fastapi import WebSocket
import engine.llm_base as llm_base


class GPT4All(llm_base.LLMBase):

    def __init__(self) -> None:
        super().__init__()

    def invoke(self, question: str, websocket: WebSocket = None):
        provider = GPT4All(
            model='./models/orca_mini_3b.gguf',
            verbose=True,
            callback_manager=CallbackManager(
                [StreamingWebsocketCallbackHandler(websocket)])
        )

        return super().invoke(question=question, llm=provider)
