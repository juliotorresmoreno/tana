import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from flow import make_pipeline
from pipe.Pipeline import Arguments

pipe = make_pipeline()

response = pipe.invoke(Arguments(question="What is anarchism?"))
# response = pipe.invoke("What are your main skills and experience as a developer?")
#response = pipe.invoke(Arguments(question="When was the war on Mars?"))
#response = pipe.invoke(Arguments(question='Who wrote "Romeo and Juliet"?'))

print("\n")
print("response: " + str(response.result) +
      ', execution_time: ' + str(response.execution_time))
