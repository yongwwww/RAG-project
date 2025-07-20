from pydantic import Field, BaseModel
from langchain_core.prompts import ChatPromptTemplate
from llm_models.agent_model import model


class GradeHallucinations(BaseModel):
    binary: str = Field(description="回答是否基于事实，取值为'yes'or'no'")

model_with_structure = model.with_structured_output(GradeHallucinations)

prompt = ChatPromptTemplate(
    [
    ('system', """
    你是一个评估生成内容是否基于检索事实的评分器。
    若回答是基于或支持于给定的事实集的，就返回"yes"；否则，请返回"no"。
    输出为json格式： 'binary': 取值为'yes'、'no'。
    """),
    ('human', '事实集：\n\n {documents} \n\n 生成内容：{generation}'),
    ]
)

#构建幻觉检测工作流
hallucination_chain = prompt | model_with_structure