from llm_models.agent_model import model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser


def generate_node(state: dict) -> dict:
    '''
    转换查询以生成更好的节点
    return 包含重述问题更新后的状态
    '''
    print(state)
    question = state['question']
    documents = state['documents']

    prompt = PromptTemplate(
        template='你是一个问答助手。请根据检索到的内容来回答问题并根据自身认知来丰富、优化并完善答案，如果不知道答案，请直接说明。回答保持简介。\n问题：{question} \n上下文：{context} \n回答：',
        input_variables=['question', 'context']
    )

    def format_docs(documents):
        if isinstance(documents, list):
            return '\n\n'.join(document.page_content for document in documents)
        else:
            return '\n\n' + documents.page_content
    
    chain = prompt | model | StrOutputParser()

    generation = chain.invoke({'question': question, 'context': format_docs(documents)})
    
    return {'question': question, 'documents': format_docs(documents), 'generation':generation}