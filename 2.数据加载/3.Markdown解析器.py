from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(
    file_path=r'/root/lanyun-tmp/RAG(Milvus)/project/datasets/performance_faq.md',
    mode='elements',                # 两种解析模式 single和 elements, single将整个Markdown作为单个文档加载不保留结构信息
    strategy='fast',
)

docs = loader.load()

print("docs的数量是:", len(docs))
for i in range(len(docs)):
    print(docs[i])
# print(docs[0].metadata)
# print(docs[0].page_content)



