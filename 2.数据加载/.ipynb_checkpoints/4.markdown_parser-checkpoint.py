from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from llm_models.embeddings_model import beg_embeddings
from langchain_experimental.text_splitter import SemanticChunker

from typing import List

class MarkdownParser:
    '''
    专门负责markdown文件的解析和切片
    '''
    def __init__(self):
        self.text_splitter = SemanticChunker(
            beg_embeddings, 
            breakpoint_threshold_type='percentile'
        )
    # document中内容太多需要进行语义切割
    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for doc in datas:
            if len(doc.page_content) > 6000:
                new_docs.extend(self.text_splitter.split_documents([doc]))
            else:
                new_docs.append(doc)
        return new_docs
    
    def parse_markdown_to_documents(
        self,
        md_file,
        encoding='utf-8'
    ) -> List[Document]:
        documents = self.parse_md(md_file)
        print(f'文件解析后的docs长度：{len(documents)}')
        merged_documents = self.merge_title_content(documents)
        print(f'合并后的docs长度：{len(merged_documents)}')
        chunk_documents = self.text_chunker(merged_documents)
        print(f'语义切割后的docs长度：{len(chunk_documents)}')
        return chunk_documents
        
    def parse_md(self, md_file: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=r'/root/lanyun-tmp/RAG(Milvus)/project/datasets/operational_faq.md',
            mode='elements',                # 两种解析模式 single和 elements, single将整个Markdown作为单个文档加载不保留结构信息
            strategy='fast',
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        return docs
        

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        merged_data = []
        parent_dict = {}    # 是一个父document的字典, 可以为当前父document的ID
        for document in datas:
            metadata = document.metadata
            parent_id = metadata.get("parent_id", None)
            category = metadata.get("category", None)
            element_id = metadata.get("element_id", None)

            if category == 'NarrativeText' and parent_id is None:  # 是否为：内容document
                merged_data.append(document)
            if category == 'Title':
                document.metadata['title'] = document.page_content
                # 如果他的父id已经存在
                if parent_id in parent_dict:
                    document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                parent_dict[element_id] = document
            if category != 'Title' and parent_id:
                parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                    

        if parent_dict is not None:
            merged_data.extend(parent_dict.values())

        return merged_data

if __name__ == "__main__":
    parase = MarkdownParser()
    docs = parase.parse_markdown_to_documents('/root/lanyun-tmp/RAG(Milvus)/project/datasets/performance_faq.md')
    # for item in docs:
    #     print(f'元数据：{item.metadata}')
    #     print(f'标题：{item.metadata.get('title', None)}')
    #     print(f'页内容：{item.page_content}\n')
    #     print("------"*10)
        