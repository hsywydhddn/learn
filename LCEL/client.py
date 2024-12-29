from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:12312/chain/")
cc=remote_chain.invoke({"language": "English", "text": "滚"})

print(cc)