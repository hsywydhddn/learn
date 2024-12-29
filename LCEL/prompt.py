from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint="https://12205-m2hl4tqk-eastus.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
    azure_deployment="gpt-35-turbo",
    openai_api_version="2024-08-01-preview",
    api_key="d7f27353b3b3463bb02b2708df922f35"
)
from langchain_core.messages import HumanMessage, SystemMessage
# 提示词模板
from langchain_core.prompts import ChatPromptTemplate
# 创建字符串，将格式化为系统消息
system_template="Translate the following into {language}"
# 创建PromptTemplate,将system_template和一个模板的组合，防止翻译的文本
prompt_template=ChatPromptTemplate.from_messages(
    [("system",system_template),("user","{text}")]
)
# 使用prmpt_template的invoke来处理请求
result=prompt_template.invoke({"language":"italian","text":"hi"})
print(result)
a=result.to_messages()
print(a)