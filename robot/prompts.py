from langchain_openai import AzureChatOpenAI
model=AzureChatOpenAI(
    azure_endpoint="https://12205-m2hl4tqk-eastus.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
    azure_deployment="gpt-35-turbo",
    openai_api_version="2024-08-01-preview",
    api_key="d7f27353b3b3463bb02b2708df922f35"
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant. Answer all questions to the best of your ability."),
    ("user", "{message}")
])
chain=prompt | model
response=chain.invoke({"message":[HumanMessage(content="hi! i'm bob")]})
a1=response.content
print(a1)
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages import  AIMessage
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
a12=trimmer.invoke(messages)
print(a12)