import json

from langchain_core.documents import Document

def load_doc_from_json(json_file):
    """从JSON文件加载为Document对象"""
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return Document(page_content=data['page_content'], metadata=data['metadata'])

# 加载之前保存的json文件
print(load_doc_from_json('./output/1_1.json'))