from knowledge.qdrant import QdrantLibrary
from knowledge.elasticsearch import ElasticSearchLibrary
from llm.text2text_generation import Text2TextGenerationModel
from llm.question_answering import QuestionAnsweringModel
from pipe.Pipeline import Pipeline
from pipe.reinformer.reinformer import Reinformer


def make_pipeline():
    chatbot = Text2TextGenerationModel()
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
    pipe.insert(Reinformer(True))

    return pipe


# response = pipe.invoke("What is anarchism?")
# response = pipe.invoke("What are your main skills and experience as a developer?")


#print("\n")
# print("response: " + str(response.result) +
#      ', execution_time: ' + str(response.execution_time))
