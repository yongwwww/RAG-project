from pydantic import Field, BaseModel
from langchain_core.prompts import ChatPromptTemplate
from llm_models.agent_model import model


class GradeAnswer(BaseModel):
    binary: str = Field(description="回答是否解决了问题，取值为'yes'or'no'")

model_with_structure = model.with_structured_output(GradeAnswer)

prompt = ChatPromptTemplate(
    [
    ('system', """
    你是一个评估生成内容是否解决了问题的评分器。
    若生成内容解决了问题，就返回"yes"；否则，请返回"no"。
    输出为json格式： 'binary': 取值为'yes'、'no'。
    """),
    ('human', '事实集：\n\n {documents} \n\n 生成内容：{generation}'),
    ]
)

#构建幻觉检测工作流
answer_grade_chain = prompt | model_with_structure