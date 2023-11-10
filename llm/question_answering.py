from langchain.llms import HuggingFacePipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, Pipeline
from decouple import config
from llm.ModelBase import ModelBase
import time
from pipe.Pipeline import Response, Arguments

CHECKPOINT = config("QUESTION_ANSWERING_MODEL")

DEVICE = config('DEVICE')
class QuestionAnsweringModel(ModelBase):
    task = "question-answering"
    pipe: Pipeline

    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__(self.task)
        self.pipe = pipeline(self.task, model=checkpoint, device=DEVICE)

    def invoke(self, args: Arguments) -> Response:
        if (args.required_task == self.task):
            if args.context != None:
                raise ValueError("context is required!")
            else:
                return

        start_time = time.time()

        paragraphs = args.context.split('\n')
        context = '\n'.join([p for p in paragraphs if len(p) > 10])

        response = self.pipe(question=args.question, context=context)

        start = context[:response['start']].rfind('\n')
        end = context[response['end']:].find('\n')

        result = context[start+1:response['end']+end]

        end_time = time.time()
        execution_time = end_time - start_time

        return Response(
            context=result,
            execution_time=execution_time,
            required_task=args.required_task
        )
