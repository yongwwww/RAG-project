from graph2.grade_chain import retriever_grader_chain
from log.log_utils import log


def grade_documents_node(state: dict) -> dict:
    
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    for doc in documents:
        score = retriever_grader_chain.invoke(
            {
                'context': doc.page_content,
                'question': question,
            }
        )
        grade = score.binary
        if grade == 'yes':
            log.info('---GRADE: 打印相关标识---')
            filtered_docs.append(doc)
        else:
            log.info('---GRADE: 打印不相关标识---')
            continue
    return {'documents': filtered_docs}
            