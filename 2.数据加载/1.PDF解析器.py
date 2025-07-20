# # 简单加载
# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader(file_path='../datasets/2023.acl-long.732.pdf')
# # 自动按页解析，每页一个Document
# docs = loader.load()

# print(len(docs))
# print(docs[0].metadata)
# print(docs[0].page_content)
# # 数据用于过滤用


import json
from IPython.core.display import HTML
from IPython.core.display_functions import display
from langchain_unstructured import UnstructuredLoader


def write_json(data, file_name):
    with open('./output/' + file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

loader = UnstructuredLoader(
    file_path='../datasets/2023.acl-long.732.pdf',
    strategy='hi_res', # 'fast' 快速解析，不含图片，这里使用告诉解析
    partition_via_api=False, # 若为True 需要api_key, 默认False
    # api_key='...'
)

docs = []

counter = 0

for doc in loader.lazy_load():
    # 每个doc都是json文件
    docs.append(doc)
    # 给每个文件命名
    json_file_name = str(doc.metadata.get('page_number')) + '_' + str(counter) + '.json'
    counter += 1
    # 变成字典类型，并写入json文件
    write_json(doc.model_dump(), json_file_name)

segments = [
    doc.metadata
    for doc in docs
    if doc.metadata.get("page_number") == 5 and doc.metadata.get("category")
]
print(f'表格数据为:')
print(segments)


# 若不用API需要构建Unstrctured环境
# 此处无权限安装，p5构建，下两个东西即可

'''
1.更新apt-get
sudo apt-get update
sudo apt upgrade
2.安装其他依赖
①.Poppler(PDF分析)
apt-get install poppler-utils
②.Tesseract
apt-get install tesseract-ocr
3.安装库
pip install "unstructured[pdf]" -i https://mirrors.aliyun.com/pypi/simple/
pip install unstructured  ??? 他会提示下载unstructured, but why? 不要用最好
pip install langchain langchain-community langchain-unstructured iPython -i https://mirrors.aliyun.com/pypi/simple/
4.配置yolo10镜像
①.安装依赖
pip install -U huggingface_hub
②.设置环境变量
linux
vi ~/.bashrc
把下面这段话放在最后source的前面一行（倒数第三行）
export HF_ENDPOINT=https://hf-mirror.com
刷新一下
source ~/.bashrc
运行1.解析器会下载yolo10模型
'''

'''
对于Markdown文件
pip install "unstructured[md]" nltk -i https://mirrors.aliyun.com/pypi/simple/
'''

'''
没有使用
③.下载模型
huggingface_cil download --resume-download gpt2 --local-dir gpt2
④.下载数据集
huggingface_cil download --repo-type dataset --resume-download wikitext --local-dir wikitext
'''