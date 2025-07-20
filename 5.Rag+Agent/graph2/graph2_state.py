from typing import TypedDict, Annotated, Literal, Optional, List
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.documents import Document


# 状态类
class State(TypedDict):
    # 消息列表
    # messages: Annotated[list[AnyMessage], add_messages]
    question: str
    transform_count: int
    generation: str
    documents: List[Document]