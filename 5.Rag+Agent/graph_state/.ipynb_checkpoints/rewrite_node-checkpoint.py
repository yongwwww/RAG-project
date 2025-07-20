from graph_state.graph_state import State
from llm_models.agent_model import model
from langchain_core.messages import HumanMessage
from utils.get_last_human_message import get_last_human_message


def rewrite_node(state: State) -> dict:
    '''
    转换查询以生成更好的节点
    return 包含重述问题更新后的状态
    '''
    messages = state['messages']
    question = get_last_human_message(messages).content
    
    msg = [
        HumanMessage(
            content=f""" \n
            不要回答原始问题，你的工作是把原始问题，转换成新的问题。
            分析输入并尝试理解潜在的语义意图/含义。
            这是原始的问题：
            \n ---------- \n
            {question}
            \n ---------- \n
            请提出一个改进后的问题："""
        )
    ]

    resp = model.invoke(msg)
    
    return {"messages": [resp]}