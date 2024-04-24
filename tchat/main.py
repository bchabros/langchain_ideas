from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

# add memory to chat whole 
# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("messages.json"), # save memory after closing it 
#     memory_key="messages",
#     return_messages=True
# )

# second option with summariazation history memory
memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"), # save memory after closing it 
    memory_key="messages",
    return_messages=True,
    llm=chat
)

# chat prompt template
prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])
