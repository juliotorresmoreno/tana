
from knowledge.qdrant import QdrantLibrary
from knowledge.elasticsearch import ElasticSearchLibrary
from llm.text2text_generation import Text2TextGenerationModel
from llm.mmlu.ollama import OllamaTextGenerationModel
from llm.text_generation import TextGenerationModel
from llm.question_answering import QuestionAnsweringModel
from pipe.Pipeline import Pipeline
from pipe.reinformer.base import Reinformer
from pipe.reinformer.hallucination import Hallucination


def make_pipeline():
    chatbot = OllamaTextGenerationModel() # Text2TextGenerationModel()
    enhance = QuestionAnsweringModel()
    knowledge_page = QdrantLibrary('root', .3)
    knowledge_wiki = ElasticSearchLibrary('wiki')

    knowledge_page_pipe = Pipeline()
    knowledge_page_pipe.insert(knowledge_page)
    knowledge_page_pipe.insert(Reinformer(False))

    knowledge_wiki_pipe = Pipeline()
    knowledge_wiki_pipe.insert(knowledge_wiki)
    knowledge_wiki_pipe.insert(enhance)
    knowledge_wiki_pipe.insert(chatbot)
    knowledge_wiki_pipe.insert(Reinformer(False))

    pipe = Pipeline()
    pipe.insert(knowledge_page_pipe)
    pipe.insert(knowledge_wiki_pipe)
    pipe.insert(Hallucination())
    pipe.insert(chatbot)
    pipe.insert(Reinformer(True))

    return pipe
