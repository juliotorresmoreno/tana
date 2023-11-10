from llm.text2text_generation import Text2TextModel
from llm.question_answering import QuestionAnsweringModel
from typing import Union
from llm.ModelBase import ModelBase


Task = Union['text2text-generation', 'question-answering']

tasks: dict[Task, any] = {
    'text2text-generation': Text2TextModel,
    'question-answering': QuestionAnsweringModel
}


def create_language_model(task: Task) -> ModelBase | None:
    if task not in tasks:
        return None

    return tasks[task]()
