from engine.ollama import OllamaLLM
from engine.gguf import GGUFLLM

def make_engine():
    return OllamaLLM()


engine = make_engine()
