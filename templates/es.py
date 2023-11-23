from langchain.prompts import ChatMessagePromptTemplate

prompt = "May the {subject} be with you"

chat_message_prompt = ChatMessagePromptTemplate.from_template(role="system", template=prompt)
chat_message_prompt.format(subject="force")

print(chat_message_prompt)