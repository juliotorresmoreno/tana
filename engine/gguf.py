
from callbacks.websocket import StreamingWebsocketCallbackHandler
from langchain.llms.llamacpp import LlamaCpp
from fastapi import WebSocket
from decouple import config
import engine.llm_base as llm_base
from os import path


GGUF_MODEL_PATH = config('GGUF_MODEL_PATH')
OLLAMA_TEXT_GENERATION_MODEL = config('GGUF_TEXT_GENERATION_MODEL')

model_path = path.join(GGUF_MODEL_PATH, OLLAMA_TEXT_GENERATION_MODEL)

class GGUFLLM(llm_base.LLMBase):

    def __init__(self) -> None:
        super().__init__()
        self.provider = LlamaCpp(
            model_path=model_path,
            temperature=0.75,
            max_tokens=512,
            top_p=1,
            verbose=True,
            callbacks=[]
        )

    def invoke(self, question: str, index_name: str = None, websocket: WebSocket = None):
        callback_manager = [StreamingWebsocketCallbackHandler(websocket)]
        return super().invoke(
            question=question,
            index_name=index_name,
            llm=self.provider,
            callbacks=callback_manager
        )
