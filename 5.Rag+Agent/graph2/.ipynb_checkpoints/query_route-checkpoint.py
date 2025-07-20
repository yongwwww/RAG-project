from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

from llm_models.agent_model import model


# 查询的动态路由； 根据用户的提问决策采取哪种检索


class RouteQuery(BaseModel):
    """将用户查询路由到最相关的数据源"""
    datasource: Literal['vectorstore', 'web_search'] = Field(..., description="根据用户的问题选择将其路由到向量知识库('vectorstore')或者网络搜索('web_search')")


# 带函数调用的LLM
structured_model_router = model.with_structured_output(RouteQuery)

system = '''你是一个擅长将用户问题路由到向量知识库或者网络搜索的专家。
向量知识库包含Milvus的相关文档。
对于这些主题的问题请使用向量知识库，其他情况请使用网络搜索。
请返回json格式，"datasource":"vectorstore" 或者 "datasource":"web_search"。'''
route_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system),
        ('human', "{question}")
    ]
)

question_route_chain = route_prompt | structured_model_router
