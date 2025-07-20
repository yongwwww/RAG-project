from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from log.log_utils import log
from llm_models.agent_model import model


def transform_query_node(state: dict) -> dict:

    log.info('---TRANSFROM QUERY---')
    question = state['question']
    documents = state['documents']
    transform_count = state.get('transform_count', 0)

    system = """作为问题重写器，你需要将输入的问题转化为更适合向量数据库检索的优化版本。
    情分析输入问题并理解其背后的语义意图，不要改变用户的意图，并且稍微简洁一些。"""

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system),
            ('human', "这是原始问题：\n\n{question} \n 请生成一个优化后的问题。")
        ]
    )

    rewrite_chain = rewrite_prompt | model | StrOutputParser()
    better_question = rewrite_chain.invoke({'question': question})
    return {'question': better_question, 'transform_count': transform_count+1}
    