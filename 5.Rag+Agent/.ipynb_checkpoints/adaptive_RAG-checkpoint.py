# 检索源很多的时候用自适应RAG（有我自修正的能力）动态路由机制来判断到底去哪儿检索
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.prompts import PromptTemplate

from graph2.graph2_state import State
from graph2.retriever_node import retriever_node
from graph2.web_search_node import web_search_node
from graph2.transform_query_node import transform_query_node
from graph2.grade_documents_node import grade_documents_node
from graph2.generate_node import generate_node
from graph2.query_route import question_route_chain
from graph2.grade_hallucinations_chain import hallucination_chain
from graph2.grade_answer_chain import answer_grade_chain
from utils.draw_graph import draw_graph
from log.log_utils import log

from typing import Literal
from pprint import pprint


graph = StateGraph(State)

# 节点构建 
graph.add_node('retriever', retriever_node)
graph.add_node('web_search', web_search_node)
graph.add_node('transform_query', transform_query_node)
graph.add_node('grade_documents', grade_documents_node)
graph.add_node('generate', generate_node)

graph.add_edge('web_search', 'generate')
graph.add_edge('transform_query', 'retriever')
graph.add_edge('retriever', 'grade_documents')

def question_router(state: dict):

    question = state['question']
    print(question)
    res = question_route_chain.invoke({'question': question})

    datasource = res.datasource

    if datasource in ['vectorstore', 'web_search']:
        log.info('---路由到'+ datasource + '---')
        return datasource
    return ''

graph.add_conditional_edges(START, question_router, {
    'vectorstore': 'retriever',
    'web_search': 'web_search',
})

def decide_to_generate(state: dict) -> str:
    log.info('---ASSESS GRADED DOCUMENTS---')
    filtered_documents = state['documents']
    transform_count = state.get('transform_count', 0)

    if not filtered_documents:
        if transform_count >= 2:
            log.info('---决策：所有文档都和问题无关，并且循环了两次，使用web_search---')
            return 'web_search'
        else:
            log.info('---决策：所有文档都和问题无关，将转换查询问题---')
            return 'transform_query'
    else:
        log.info('---决策：生成最终回答---')
        return 'generate'

graph.add_conditional_edges('grade_documents', decide_to_generate, {
    'transform_query': 'transform_query', 
    'generate': 'generate', 
    'web_search': 'web_search',
})

def grade_generation_and_question(state: dict) -> str:
    log.info('---检查内容生成是否存在幻觉---')
    question = state['question']
    documents = state['documents']
    generation = state['generation']

    # 检查生成是否基于文档
    score = hallucination_chain.invoke(
        {
            'documents':documents, 
            'generation':generation,
            
        }
    )

    grade = score.binary

    if grade == 'yes':
        log.info('---判定：生成的内容基于参考文档')
        # 检查是否准确回答问题
        log.info('---评估：生成回答于问题的匹配度')
        score = answer_grade_chain.invoke(
            {
                'documents':documents, 
                'generation':generation,
                
            }
        )
        grade = score.binary
        if grade == 'yes':
            log.info('---判定：生成的内容准确回答问题---')
            return 'useful'
        else:
            log.info('---判定：生成内容未能准确回答问题---')
            return 'not useful'
    else:
        print('---判定：生成的内容未基于参考文档，将重新尝试---')
        return 'not supported'

graph.add_conditional_edges(
    'generate', 
    grade_generation_and_question, 
    {
        'useful': END, 
        'not useful': 'transform_query', 
        'not supported': 'generate', 
    }
)

graph = graph.compile()

# draw_graph(graph, 'adaptive.png')

while True:
    user_input = input('用户：')
    if user_input.lower() in ['q', 'exit', 'quit']:
        print('结束对话，拜拜')
        break
    else:
        events = graph.stream({'question': user_input}, stream_mode="values")
        for event in events:
            for key, value in event.items():
                # 打印当前节点信息
                pprint(f"Node '{key}'：")
                # 可选：打印每个节点的完整状态信息
                # pprint(value, indent=2, width=80, depth=None)

            pprint('\n------------\n')
        pprint(event['generation'])



