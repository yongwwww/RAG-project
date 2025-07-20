from pymilvus import MilvusClient, Function, FunctionType, DataType
import numpy as np

# 连接服务器
client = MilvusClient(uri='http://47.111.95.149:19530')

# 定义 collection的模式，真正使用的时候不用，做测试
schema = client.create_schema()
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True, analyzer_params={"type":"Chinese")
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

# 进行稀疏嵌入的函数，从一个字段中读取原始数据，通过bm25算法，转化为稀疏向量然后存到另一个字段
bm25_function = Function(
    name="text_bm25_emb",
    input_field_names=["text"],          # 读取原始数据
    output_field_names=['sparse'],       # 存到新的字段
    function_type=FunctionType.BM25,
)

# BM25函数加入schema
schema.add_function(bm25_function)

# 配置索引（做优化的）
# 初始化索引
index_params = client.prepare_index_params()

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

# 创建一个表
client.create_collection(
    collection_name='t_demo2',
    schema=schema,
    index_params=index_params
)

# 5. 插入数据（注意数据格式要和 schema 匹配，补充主键相关逻辑，因上面设为 auto_id，插入时可不用显式指定 id ）
insert_data = [
    {'text': 'information retrieval is a field of study.'},
    {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
    {'text': 'data mining and information retrieval overlap in research.'},
]

insert = client.insert('t_demo2', insert_data)
# print(insert)

# 6. 执行检索
search_params = {
    'params': {'drop_ratio_search': 0.1},  # 检索时忽略 20% 小向量值，加速检索
}

results = client.search(
    collection_name='t_demo2',
    data=['whats the focus of information retrieval?'],  # 检索的文本，需转成稀疏向量（需补充转换逻辑！）
    anns_field='sparse',                                 # 基于 sparse 字段检索
    limit=3,                                             # 返回最相关的 3 条结果
    search_params=search_params,
    output_fields=['text']
)

print(results)