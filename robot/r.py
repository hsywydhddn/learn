from langchain_openai import AzureChatOpenAI
model=AzureChatOpenAI(
    azure_endpoint="https://12205-m2hl4tqk-eastus.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
    azure_deployment="gpt-35-turbo",
    openai_api_version="2024-08-01-preview",
    api_key="d7f27353b3b3463bb02b2708df922f35"
)
# from langchain_core.messages import HumanMessage  #ai生成的回复内容
# a1=model.invoke([HumanMessage(content="Hi! I'm five")])
# # print(a1)
# a2=model.invoke([HumanMessage(content="what's my name")])
# # print(a2)
# from langchain_core.messages import AIMessage
# a3=model.invoke(
#     [
#         HumanMessage(content="Hi,I'm ed"),
#         AIMessage(content="Hello ed! How can I assist you today"),
#         HumanMessage(content="what's my name?")
#     ]
# )
# # print(a3)
# from langchain_core.chat_history import(
#    BaseChatMessageHistory,
#    InMemoryChatMessageHistory,
# )
# #basechatrmessagehistory是定义了聊天的基本接口,
# # Inmeorychatmessaghehistory实现了在内存中存储聊天历史记录的逻辑
# from langchain_core.runnables.history import RunnableWithMessageHistory
# #runnablewithmessagehistory是用来读取和更新聊天历史
# store={}
# def get_session_history(session_id: str)-> BaseChatMessageHistory:
# # 获取会话历史的函数，如果会话ID不存在，则创建一个新的InMemoryChatMessageHistory实例
#     if session_id not in store:
#         store[session_id]=InMemoryChatMessageHistory()
#     return store[session_id]
# with_message_history=RunnableWithMessageHistory(model,get_session_history)
# # 创建RunnableWithMessageHistory实例，传入模型和获取会话历史的函数
# print(with_message_history)
# config={"configurable":{"session_id":"abc2"}}
# response=with_message_history.invoke(
#     [HumanMessage(content="Hi!I'm five")],
#     config=config,
# )
# a4=response.content#对runnablewithmessagehistory的invoke方法的相应内容的引用
# # print(a4)
# response=with_message_history.invoke(
#     [HumanMessage(content="what's my name?")],
#     config=config,
# )
# a5=response.content
# # print(a5)
# config={"configurable":{"session_id":"abc2"}}
# response=with_message_history.invoke(
#     [HumanMessage(content="what's my name")],
#     config=config,
# )
# a6=response.content
# # print(a6)
# from langchain_core.messages import HumanMessage
# from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
# #chatpromtemplate用于生成与模型对象化式交互的提示模板
# #messagesplaceholder是在插头promtemplate中插入动态消息的占位符
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "you are a helpful assistant. Answer all questions to the best of your ability."),
#     ("user", "{message}")
# ])
# chain=prompt | model
# response=chain.invoke({"message":[HumanMessage(content="hi! i'm bob")]})
# a7=response.content
# # print(a7)
# with_message_hitory=RunnableWithMessageHistory(chain,get_session_history)#runnablewithmessagehistory用来屌用runnable对象之前加载的消息历史
# config={"configurable":{"session_id":"adc5"}}
# response=with_message_hitory.invoke(
#     [HumanMessage(content="Hi! i'm jim")],
#     config=config,
# )
# a8=response.content
# # print(a8)
# response=with_message_history.invoke(
#     [HumanMessage(content="what's my name")],
#     config=config,
# )
# a9=response.content
# # print(a9)
# prompt=ChatPromptTemplate.from_messages(
#             [
#                 ("system",
#             "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
# chain=prompt | model
# response=chain.invoke(
#     {"messages":[HumanMessage(content="hi! i'm bob")],"language":"ja"}
# )
# a10=response.content
# # print(a10)
# with_message_history=RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="messages"
# )
# config={"configurable":{"session_id":"abc11"}}
# response=with_message_history.invoke(
#     {"messages":[HumanMessage(content="hi! i'm told")],"language":"Spanish"},
#     config=config,
# )
# a11=response.content
# # print(a11)
# response=with_message_history.invoke(
#     {"messages":[HumanMessage(content="what's my name ")],"language":"Chanese"},
#     config=config,
# )
# a12=response.content
# print(a12)
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages import  AIMessage
from typing import List

# pip install tiktoken
import tiktoken
from langchain_core.messages import BaseMessage, ToolMessage


def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """Approximately reproduce https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    For simplicity only supports str Message.contents.
    """
    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
            tokens_per_message
            + str_token_counter(role)
            + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


trimmer = trim_messages(
    max_tokens=35,
    strategy="last",
    token_counter=tiktoken_counter,
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
print(trimmer.invoke(messages))
# a13=trimmer.invoke(messages)
# print(a13)
# from operator import itemgetter
#
# from langchain_core.runnables import RunnablePassthrough
#
# chain = (
#     RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
#     | prompt
#     | model
# )
#
# response = chain.invoke(
#     {
#         "messages": messages + [HumanMessage(content="what's my name?")],
#         "language": "English",
#     }
# )
# b1=response.content
# print(b1)
# response = chain.invoke(
#     {
#         "messages": messages + [HumanMessage(content="what math problem did i ask")],
#         "language": "English",
#     }
# )
# b2=response.content
# print(b2)