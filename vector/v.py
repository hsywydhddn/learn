from langchain_openai import AzureChatOpenAI
modle=AzureChatOpenAI(
    azure_endpoint="https://arco-swedencentral.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
    azure_deployment="gpt-35-turbo",
    openai_api_version="2024-08-01-preview",
    api_key="e7a33c79acc54e019fee4ddb146796de"
)
#创建一个包含文档的向量数据库，并从中查询关于与cat最相似的文档
from langchain_core.documents import Document
documents=[
    Document(
        page_content="dogs are great companions, knows for their loyalty and friendliness",
        metadata={"source":"mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source":"mammal-pets-doc"},
    ),
    Document(
        page_content="goldfish are popular pets for beginners,requiring relatively simple care",
        metadata={"source":"fish-pets-doc"}

    ),
    Document(
        page_content="parrots are intelligent birds capable of mimicking human speech",
        metadata={"source":"bird-pets-doc"}

    ),
    Document(
        page_content="rabbits are social animals that need plenty of space to hop around,.",
        metadata={"source":"mammal-pets-doc"}
    ),
]
from langchain_chroma import Chroma#chroma向量数据库,用来存储和查询向量嵌入
from langchain_openai import OpenAIEmbeddings#openaiembeddings用于生成文档的向量嵌入
vectorstore=Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)
a1=vectorstore.similarity_search("cat")
print(a1)
#输出结果为什么是这样？#为什么结果不包括、goldfish
#[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
# Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
# Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
# Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
#异步查询
a2=await vectorstore.similarity_search("cat")#暂停当前协议的执行，知道等待的异步操作完成。发送一个异步的相似度搜索请求，查询与字符串‘cat’最相似的文档，暂停当前协议的执行，知道搜索完成并返回结果
print(a2)
# [Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
#  Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
#  Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
#  Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
#返回分数
a3=vectorstore.similarity_search_with_score("cat")
print(a3)
# [(Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
#   0.3751849830150604),
#  (Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
#   0.48316916823387146),
#  (Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
#   0.49601367115974426),
#  (Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'}),
#   0.4972994923591614)]
#根据与嵌入查询的相似性返回文档
embedding=OpenAIEmbeddings().embed_query("cat")#embed_query接受字符串作为变量并且返回该字符串的嵌入向量
a4=vectorstore.similarity_search_by_vector(embedding)#执行相似度搜索
print(a4)
# [Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
#  Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
#  Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
#  Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
from langchain_core.runnables import RunnableLambda
retriver=RunnableLambda(vectorstore.similarity_search).bind(k=1)#k=1代表在进行相似度搜索时，你希望返回最相似的前k个结果，bind方法时创建一个新的函数或对象通过将一个或者多个参数绑定到函数上。
retriver.batch(["cat","shark"])#batch方法用来对两个查询词进行相似度搜索
# [[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'})],
#  [Document(page_content='Goldfish are popular pets for beginners, requiring relatively simple care.', metadata={'source': 'fish-pets-doc'})]]
retriver=vectorstore.as_retriever(   #使用as_retriever方法生成一个检索器
    search_type="similaruty",
    search_kwargs={"k":1},
)
retriver.batch(["cat","shark"])
# [[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'})],
#  [Document(page_content='Goldfish are popular pets for beginners, requiring relatively simple care.', metadata={'source': 'fish-pets-doc'})]]
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough
message="""
Answer this question using the provided context only.
{question}
Context:
{context}
"""

prompt=ChatPromptTemplate.from_messages(["human",message])
#构建RAG链
rag_chain={"context":retriver,"question":RunnablePassthrough()} | prompt | modle#runnablepassthrough在链中起到了透明传递的作用，确保不会对输入数据进行任何修改。而是直接将其传递给下一个环节。
response=rag_chain.invoke("tell me about cats")#在向量数据库中查找关于cats的信息
print(response.content)
#Cats are independent pets that often enjoy their own space.