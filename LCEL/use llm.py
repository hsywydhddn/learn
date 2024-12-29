import getpass
import os

from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint="https://12205-m2hl4tqk-eastus.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
    azure_deployment="gpt-35-turbo",
    openai_api_version="2024-08-01-preview",
    api_key="d7f27353b3b3463bb02b2708df922f35"
)

from langchain_core.messages import HumanMessage, SystemMessage

messages1 = [
    SystemMessage(content="请用中文"),#系统消息，指引ai的行为
    HumanMessage(content="输出一个笑话"),#用户消息
]
a1=model.invoke(messages1)#用模型的invoke来处理消息列表
print(a1)
messages2 = [
    SystemMessage(content="将下列的英语转化为日语"),#系统消息
    HumanMessage(content="hi"),#用户消息
]
from langchain_core.output_parsers import StrOutputParser
#创建StOutputParser来解析模型的输出,因为模型的相应可能包含多种数据格式，StOutputParser可以将这些相应转化为字符串格式
parser= StrOutputParser()
result= model.invoke(messages2)
a2=parser.invoke(result)
print(a2)
#用|创建链，将模型与输出解释器链式连接
chain = model | parser#|在langchain中将两个元素合并
a3=chain.invoke(messages1)
print(a3)