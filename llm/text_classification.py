from langchain.llms import HuggingFacePipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, Pipeline
from decouple import config
from llm.ModelBase import ModelBase
from pipe.Pipeline import Arguments, Response


CHECKPOINT = config("TEXT_CLASSIFICATION")


class TextClassificationModel(ModelBase):
    task = "text-classification"
    pipe: Pipeline
    hf: HuggingFacePipeline

    def __init__(self, checkpoint: str = CHECKPOINT):
        super().__init__(self.task)
        self.pipe = pipeline(
            self.task,
            model=checkpoint,
        )
        self.hf = HuggingFacePipeline(
            pipeline=self.pipe
        )

    def invoke(self, args: Arguments) -> Response:
        pass
