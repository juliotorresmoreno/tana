from langchain.llms import Ollama
from llm.ModelBase import ModelBase
import pandas as pd
from pipe.Pipeline import Response, Arguments
from decouple import config
from langchain.prompts import PromptTemplate
from constants import RESPOND_BASED_ON_CONTEXT, RESPOND_BASED_ON_CHAT
import time

OLLAMA_URL = config('OLLAMA_URL')

CHECKPOINT = config("OLLAMA_TEXT_GENERATION_MODEL")

tasks = pd.read_json('templates/text-generation.json', orient='table')


class OllamaTextGenerationModel(ModelBase):
    task = 'text-generation'
    default_template: str
    engine: Ollama

    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__(self.task)
        self.engine = Ollama(base_url=OLLAMA_URL, model=checkpoint)
        self.default_template = tasks.template[tasks['task'] == self.task][0]

    def invoke(self, args: Arguments) -> Response:
        if args.required_task == RESPOND_BASED_ON_CONTEXT:
            if args.context != None:
                return self.generate(args.required_task, {
                    "question": args.question,
                    "context": args.context
                })
            else:
                return Response(execution_time=0)

        if args.required_task == RESPOND_BASED_ON_CHAT:
            return self.generate(args.required_task, {"question": args.question})

        return self.generate(args.required_task, {"question": args.question})

    def generate(self, task: str = 'text-generation', args: dict = {}):
        start_time = time.time()
        templates = tasks.template[tasks['task'] == task]

        template = ''
        if templates.shape[0] > 0:
            template = templates.iloc[0]
        else:
            template = self.default_template

        prompt = PromptTemplate.from_template(template).format(**args)

        result = self.engine(prompt)

        end_time = time.time()
        execution_time = end_time - start_time

        return Response(release=result, execution_time=execution_time)
