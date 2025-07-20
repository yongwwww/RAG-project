#deepseek-v3,qwen满血版,gpt4-4o,clausd-3.5,
#调用智普AI 实现聊天机器人案列
import os

from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.messages import HumanMessage
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from milvusvector import MilvusVectorSave

# 创建模型
model = ChatOpenAI(
    model='glm-4-plus',
    temperature='1',
    api_key='8abe4511793c486c8026c94faab71e3f.OW2GRGvIlhAkv7pR',
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手，判断是否需要调用工具，并尽可能的回答问题.'),
    MessagesPlaceholder(variable_name='messages', optional=True)              # optional未来这个参数可以不给
])

# mvs = MilvusVectorSave()
# mvs.create_connection()

# retriever = mvs.vector_store_saved.as_retriever(
#     search_type='similarity',   # 仅返回相似度超过阈值的
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.1,
#         "ranker_type": "rrf",
#         "ranker_params": {"k": 100},
#         "filter": {"category": "content"}
#     }
# )

# retriever_tool = create_retriever_tool(
#     retriever,
#     name="rag_retriever",
#     description='搜索并返回关于“milvus”的信息，包括：“操作，表现，产品”等信息'
# )

# chain = prompt | model.bind_tools([retriever_tool])

# print(chain.invoke({'messages':[('user', 'how to use milvus?')]}))

from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import initialize_agent, AgentType


# 1. 修复连接和检索器配置
mvs = MilvusVectorSave()
mvs.create_connection()

retriever = mvs.vector_store_saved.as_retriever(
    search_type='similarity',
    search_kwargs={
        "k": 3,
        "score_threshold": 0.2,  # 提高阈值，避免返回低相关结果
        "ranker_type": "rrf",  # 暂时移除，similarity模式下可能不支持
        "ranker_params": {"k": 100},
        "filter": {"category": "content"}  # 修复拼写错误
    }
)

# 2. 创建检索工具
retriever_tool = create_retriever_tool(
    retriever,
    name="rag_retriever",
    description='搜索并返回关于“milvus”的信息，包括：“操作，表现，产品”等信息'
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你有一个工具可以查询Milvus相关信息，请根据用户问题决定是否使用它。"),
    ("user", "{input}")
])


# 第三方工具需要pip install numexpr arxiv
tools = [retriever_tool]

agent = initialize_agent(
    tools,
    model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

print(agent.invoke('Milvus on Windows'))