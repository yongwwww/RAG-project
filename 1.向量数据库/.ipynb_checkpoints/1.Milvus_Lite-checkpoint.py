from pymilvus import MilvusClient
import numpy as np

# 初始化Milvus，并指定存放路径
client = MilvusClient('./milvus_demo.db')

# 集合类似于关系型数据库的表，用于存储向量和其他字段,建立连接
client.create_collection(
    collection_name='demo_collection',   #表名
    dimension=384
)
# 准备数据：文档、向量以及其他字段
docs = [
    'Artificial intelligence was founded as an academic discipline in 1956.', 
    'Alan Turing was the first person to conduct substantial research in AI.',
    'Born in Maida Vale, London, Turing was raised in southern England.'
]
# 生成随机向量（应该用embeddings代替）
vectors = [[np.random.uniform(-1, 1) for _ in range(384)] for _ in range( len (docs))]

# 将文档、向量、ID和主题打包成字典格式
# 每个字典包含以下字段：
'''
id: 唯一标识符
vector: 向量数据
text: 文本数据
subject: 主题标签
'''
data = [
    {'id': i, 'vector': vectors[i], 'text': docs[i], 'subject': 'history'}
    for i in range(len(vectors))
]

# 将数据插入到集合中， res是插入结果
res = client.insert(
    collection_name='demo_collection',
    data=data
)

print('Insert:', res, end='\n\n')

# 检索向量最相似的结果
res = client.search(
    collection_name='demo_collection',
    data=[vectors[0]], # 查询向量
    filter="subject == 'history'",
    limit=2,
    output_fields=['text', 'subject']#指定返回的字段
)

print('Search:', res, end='\n\n')

# 过滤搜索，条件搜索
# 类似以SQL语句
res = client.query(
    collection_name='demo_collection',
    filter="subject == 'history'",
    output_fields=['text', 'subject']
)

print('Query:', res, end='\n\n')

res = client.delete(
    collection_name='demo_collection',
    filter="subject == 'history'",
)

print('Delete:', res, end='\n\n')
