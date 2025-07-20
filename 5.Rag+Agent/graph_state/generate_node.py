from graph_state.graph_state import State, grade
from llm_models.agent_model import model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from utils.get_last_human_message import get_last_human_message
from langchain.schema.output_parser import StrOutputParser


def generate_node(state: State) -> dict:
    '''
    转换查询以生成更好的节点
    return 包含重述问题更新后的状态
    '''
    messages = state['messages']
    question = get_last_human_message(messages).content
    last_message = messages[-1]

    docs = last_message.content

    prompt = PromptTemplate(
        template='你是一个问答助手。请根据检索到的内容来回答问题并根据自身认知来丰富、优化并完善答案，如果不知道答案，请直接说明。回答保持简介。\n问题：{question} \n上下文：{context} \n回答：',
        input_variables=['question', 'context']
    )
    
    chain = prompt | model | StrOutputParser()

    resp = chain.invoke({'question': question, 'context': docs})
    ai_message = AIMessage(content=resp)
    return {'messages': [ai_message]}