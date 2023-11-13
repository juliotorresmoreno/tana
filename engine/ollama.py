
from callbacks.realtime import StreamingWebsocketCallbackHandler
from langchain.llms import Ollama
from fastapi import WebSocket
from decouple import config
import engine.llm_base as llm_base

OLLAMA_TEXT_GENERATION_MODEL = config('OLLAMA_TEXT_GENERATION_MODEL')


class OllamaLLM(llm_base.LLMBase):

    def __init__(self) -> None:
        super().__init__()
        self.provider = Ollama(
            model=OLLAMA_TEXT_GENERATION_MODEL,
            verbose=False,
            top_p=.9,
            top_k=40,
            temperature=.9
        )

    def invoke(self, question: str, index_name: str = None, websocket: WebSocket = None):
        callback_manager = [StreamingWebsocketCallbackHandler(websocket)]
        return super().invoke(
            question=question,
            index_name=index_name,
            llm=self.provider,
            callbacks=callback_manager
        )
