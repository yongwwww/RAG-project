from graph_state.agent_node import retriever
from log.log_utils import log


def retriever_node(state: dict) -> dict:
    log.info('---去知识库中检索文档---')
    question = state['question']
    print(question)

    documents = retriever.invoke(question)

    return {'documents': documents}