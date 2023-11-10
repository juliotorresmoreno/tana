from langchain.llms import HuggingFacePipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, Pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from decouple import config
from llm.ModelBase import ModelBase
import time
import pandas as pd
from pipe.Pipeline import Response, Arguments
from constants import RESPOND_BASED_ON_CONTEXT, RESPOND_BASED_ON_CHAT

CHECKPOINT = config("TEXT_GENERATION_MODEL")

tasks = pd.read_json('templates/text-generation.json', orient='table')


class TextGenerationModel(ModelBase):
    pipe: Pipeline
    hf: HuggingFacePipeline
    task = 'text-generation'
    default_template: str

    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__(self.task)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint)

        self.pipe = pipeline(
            self.task,
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=512
        )
        self.hf = HuggingFacePipeline(pipeline=self.pipe)
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

        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.hf

        result = chain.invoke(args)

        end_time = time.time()
        execution_time = end_time - start_time

        return Response(release=result, execution_time=execution_time)
