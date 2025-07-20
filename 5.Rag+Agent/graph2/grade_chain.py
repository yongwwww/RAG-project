from langchain_core.prompts import PromptTemplate
from typing import Literal
from utils.get_last_human_message import get_last_human_message
from llm_models.agent_model import model
from pydantic import BaseModel, Field


class grade(BaseModel):
    '''相关性的二元评分'''
    binary: str = Field(description="文档是否与问题相关：'yes' or 'no'")

model_with_structure = model.with_structured_output(grade)

prompt = PromptTemplate(
    template="""
    你是一个文档相关性评估专家。请根据以下提供的检索内容和用户问题，判断检索内容是否足够回答用户的问题。
    
    评估标准：
    1. 相关性：检索内容是否包含与问题直接相关的信息？
    2. 准确性：检索内容是否准确，不存在冲突或错误？
    
    用户问题：{question}
    检索到的内容：{context}
    
    只要有相关性，并且无逻辑错误，就返回"yes"；否则，请返回"no"。
    请返回json格式，'binary':'yes' 或者 'binary':'no'。
    """,
    input_variables=['context', 'question']
)
retriever_grader_chain = prompt | model_with_structure