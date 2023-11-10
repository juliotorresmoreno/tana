from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = OpenAI(openai_api_key=OPENAI_API_KEY)



llm = OpenAI()
chat_model = ChatOpenAI()

#llm.predict("hi!")
chat_model.predict("hi!")
