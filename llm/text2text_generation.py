from langchain.llms import HuggingFacePipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, Pipeline
from langchain.prompts import PromptTemplate
from decouple import config
from llm.ModelBase import ModelBase
import time
import pandas as pd
from pipe.Pipeline import Response

CHECKPOINT = config("TEXT2TEXT_GENERATION_MODEL")

tasks = pd.read_json('templates/text2text-generation.json', orient='table')


class Text2TextGenerationModel(ModelBase):
    pipe: Pipeline
    hf: HuggingFacePipeline
    task = 'text2text-generation'
    default_template: str

    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__(self.task)
        self.pipe = pipeline(self.task, model=checkpoint, max_new_tokens=512)
        self.hf = HuggingFacePipeline(
            pipeline=self.pipe
        )
        self.default_template = tasks.template[tasks['task'] == self.task][0]

    def invoke(self, question: str, context: str | None = None, task: str | None = None) -> Response:        
        if task == 'text2text-generation-from-context':
            if context != None:
                return self.generate(task, {"question": question, "context": context})
            else:
                return Response(execution_time=0)

        if task == 'text2text-generation-from-chat':
            return self.generate(task, {"question": question})

        return self.generate(task, {"question": question})

    def generate(self, task: str = 'text2text-generation', args: dict = {}):
        start_time = time.time()
        templates = tasks.template[tasks['task'] == task]

        template = ''
        if templates.shape[0] > 0:
            template = templates.iloc[0]
        else:
            template = self.default_template

        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.hf

        result = chain.invoke(args)

        end_time = time.time()
        execution_time = end_time - start_time

        return Response(result=result, execution_time=execution_time)
