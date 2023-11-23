from llama_cpp import Llama
llm = Llama(model_path="/home/juliotorres/models/llama2-7b-q4_K_M.bin")
output = llm("Q: what is cheese? A: ", max_tokens=512, stop=["Q:", "\n"], echo=True)

print(output)
