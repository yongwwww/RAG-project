from langchain_core.documents import Document
from log.log_utils import log
from tools.web_search import web_search_tool


def web_search_node(state: dict) -> dict:
    log.info('---WEB SEARCH---')
    question = state['question']

    docs = web_search_tool.invoke({'query': question})['results']
    
    web_results = '\n'.join([doc['content'] for doc in docs])
    web_results = Document(page_content=web_results)

    return {'documents': web_results}