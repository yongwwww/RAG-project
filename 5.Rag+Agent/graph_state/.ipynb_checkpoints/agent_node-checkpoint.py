from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from graph_state.graph_state import State
from llm_models.agent_model import model
from graph_state.milvusvector import MilvusVectorSave


# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手，尽量的调用工具，并尽可能的回答问题.'),
    ("placeholder", "{messages}")              # optional未来这个参数可以不给
])

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
    description='用于检索关于“Milvus”的信息，包括：Milvus的“操作，表现，产品”等信息'
)


def agent_node(state: State):
    messages = state['messages']
    
    runnable = prompt | model.bind_tools([retriever_tool])
    resp = runnable.invoke({'messages': messages})
    
    return {"messages": [resp]}