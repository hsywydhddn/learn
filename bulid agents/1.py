from langchain_openai import AzureChatOpenAI
model=AzureChatOpenAI(
azure_endpoint="https://12205-m2hl4tqk-eastus.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
    azure_deployment="gpt-35-turbo",
    openai_api_version="2024-08-01-preview",
    api_key="d7f27353b3b3463bb02b2708df922f35"
)
import getpass
import os

tavily_api_key = os.getenv('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')
from langchain_community.tools.tavily_search import TavilySearchResults
search=TavilySearchResults(max_results=2)
search_results=search.invoke("what is the weather in SF")
print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]
print(tools)

























