from engine.ollama import invoke
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


response, time = invoke(
    'What are the various approaches to Task Decomposition for AI Agents?')

print(response)
print('time: ' + str(time))
