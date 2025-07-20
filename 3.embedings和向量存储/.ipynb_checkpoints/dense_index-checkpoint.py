from langchain_core.documents import Document
from typing import List
from pymilvus import MilvusClient, Function, FunctionType, DataType, IndexType
from pymilvus.client.types import MetricType
from langchain_milvus import Milvus
from utils.env import MILVUS_URI, COLLECTION_NAME
from llm_models.embeddings_model import beg_embeddings
from langchain_milvus import BM25BuiltInFunction
from documents.markdown_parser import MarkdownParser



class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self):
        """自定义collection的索引"""
        self.vector_store_saved: Milvus= None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URI)

        schema = client.create_schema()
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True, analyzer_params={"tokenizer":"jieba", "filter":["cnalphanumonly"]})
        # 一定要先查看数据中有些什么字段，所创建的表，必须接收到该字段。
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="filetype", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000, nullable=True) 
        schema.add_field(field_name="category_depth", datatype=DataType.INT64, nullable=True)

        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],          # 读取原始数据
            output_field_names=['sparse'],       # 存到新的字段
            function_type=FunctionType.BM25,
        )

        schema.add_function(bm25_function)

        index_params = client.prepare_index_params()

        # 密集向量字段
        index_params.add_index(
            field_name="dense",
            index_name="dense_vector_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors 固定的
            metric_type=MetricType.IP,                  # 固定的
            params={
                "M": 32,            # 紧邻的节点数，就是一个点最多可以连接多少个相似的点，越多精度越高，但占用内存，速度慢，64最大。
                "efConstruction":64 # 搜索范围 50-200，按照数据集规模来定，数据越大搜索范围越大，才能保证精度，
                },
        )

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors 固定的
            metric_type="BM25",                  # 固定的
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",  # 构建索引的算法，优先查出来评分最高的。    还有DAAT_WAND
                "bm25_k1": 1.6,                          # 范围 [1.2,2.0]   文档中术语的重要程度
                "bm25_b": 0.75                           # 归一化程度，0表示完成归一化，稀疏向量不怎么需要归一化
            },
        )
        
        # 判断collection是否存在
        if COLLECTION_NAME in client.list_collections():
            # 先释放（删除数据）和索引再删除collection
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(
                collection_name=COLLECTION_NAME, 
                index_name="dense_vector_index"
            )
            client.drop_index(
                collection_name=COLLECTION_NAME, 
                index_name="sparse_inverted_index"
            )
            client.drop_collection(collection_name=COLLECTION_NAME)
        # 创建一个表
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
    def create_connection(self):
        """创建一个连接，milvus+langchain"""
        # 初始化langchain的Milvus
        self.vector_store_saved = Milvus(
            embedding_function=[beg_embeddings, ],            # 可以接多个
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(), 
            vector_field=["dense", "sparse"],               # 密集向量要在稀疏向量前面
            consistency_level="Strong",                      # picture2 一般使用strong和Session
            auto_id=True,
            connection_args={
                'uri': MILVUS_URI
            }
        )
        

    def add_documents(self, datas: List[Document]):
        """把新的document导入到Milvus中"""
        self.vector_store_saved.add_documents(datas)


if __name__ == "__main__":
    filepath = '/root/lanyun-tmp/RAG(Milvus)/project/datasets/operational_faq.md'
    parase = MarkdownParser()
    docs = parase.parse_markdown_to_documents(filepath)

    mvs = MilvusVectorSave()
    mvs.create_collection()
    mvs.create_connection()
    mvs.add_documents(docs)

    client = mvs.vector_store_saved.client
    # 得到表结构
    desc = client.describe_collection(collection_name=COLLECTION_NAME)

    print(f'表结构是：{desc}')

    res = client.list_indexes(collection_name=COLLECTION_NAME)
    print('表中的所有索引：', res)

    if res:
        for i in res:
            # 得到索引描述
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print(desc_index)
