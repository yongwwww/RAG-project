from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# 状态类
class State(TypedDict):
    # 消息列表
    messages: Annotated[list[AnyMessage], add_messages]

class grade(BaseModel):
    '''相关性的二元评分'''
    binary: str = Field(description="相关性评分，可选值：'yes' or 'no'")