from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    model='qwen-plus',
    temperature=0.75,
    api_key='sk-54852cf1705f4dada0cb8343804578b0',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)