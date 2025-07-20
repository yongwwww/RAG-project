from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.prompts import PromptTemplate

from graph_state.agent_node import agent_node, retriever_tool
from graph_state.rewrite_node import rewrite_node
from graph_state.generate_node import generate_node
from graph_state.graph_state import State, grade
from llm_models.agent_model import model
from utils.draw_graph import draw_graph
from utils.get_last_human_message import get_last_human_message

from typing import Literal


graph = StateGraph(State)

# 节点构建    
graph.add_node('agent', agent_node)
graph.add_node('rewrite', rewrite_node)
graph.add_node('generate', generate_node)

tool_node = ToolNode(tools=[retriever_tool])
graph.add_node('retriever', tool_node)

# 边构建
graph.add_edge(START, 'agent')
graph.add_conditional_edges('agent', tools_condition,{
    'tools': 'retriever',
    END: END
})

def grade_document(state: State) -> Literal['generate', 'rewrite']:
    '''
    检查生成的相关性
    '''
    model_with_structure = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""
        你是一个文档相关性评估专家。请根据以下提供的检索内容和用户问题，判断检索内容是否足够回答用户的问题。
        
        评估标准：
        1. 相关性：检索内容是否包含与问题直接相关的信息？
        2. 准确性：检索内容是否准确，不存在冲突或错误？
        
        用户问题：{question}
        检索到的内容：{context}
        
        只要有相关性，并且无逻辑错误，就返回'binary':"yes"；否则，请返回'binary':"no"。注意：根据结构优化输出来返回json格式
        """,
        input_variables=['context', 'question']
    )
    chain = prompt | model_with_structure

    question = get_last_human_message(state['messages']).content

    docs = state['messages'][-1].content

    score = chain.invoke({'question': question, 'context': docs})
    print(score.type)

    if score.binary == 'yes':
        return 'generate'
    return 'rewrite'
    
graph.add_conditional_edges('retriever', grade_document)
graph.add_edge('rewrite', 'agent')
graph.add_edge('generate', END)

memory = MemorySaver()
graph = graph.compile(checkpointer=memory)

draw_graph(graph, 'corrective_RAG.png')

import uuid

session_id = str(uuid.uuid4())
config = {'configurable':{"thread_id":session_id}}

while True:
    user_input = input('用户：')
    if user_input.lower() in ['q', 'exit', 'quit']:
        print('结束对话，拜拜')
        break
    else:
        events = graph.stream({'messages':[('user', user_input)]}, config, stream_mode="values")
        for event in events:
            if 'messages' in event:
                event["messages"][-1].pretty_print()
