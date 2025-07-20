import multiprocessing
from multiprocessing import Queue
from dense_index import MilvusVectorSave
import os
from documents.markdown_parser import MarkdownParser
# 采用分布式，多进程的方式把海量数据写入Milvus数据库
'''
使用以下指令，安装milvus客户端，查看数据
docker run -d -p 8000:3000 -e MILVUS_URL=公网IP:19530 zilliz/attu:v2.5
然后使用http://公网IP:8000来访问milvus客户端
'''

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int=20):
    # 进程1：解析目录下所有md文件并分批放入队列
    print(f"解析进程开始扫描目录：{dir_path}")

    # 获取目录下所有.md文件
    md_files=[
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.md')
    ]

    if not md_files:
        print("warnning: Not found markdown file")
        output_queue.put(None)
        return
    parser = MarkdownParser()
    doc_batch = []
    for file_path in md_files:
        try:
            docs = parser.parse_markdown_to_documents(file_path)
            if docs:
                doc_batch.extend(docs)
            # 达到批次大小放入队列中
            if len(doc_batch) >= batch_size:
                # 放入队列
                output_queue.put(doc_batch.copy())
                # 清空当前缓冲区所有数据
                doc_batch.clear()
        except Exception as e:
            print(f"解析失败 {file_path}: {str(e)}")
    if doc_batch:
        output_queue.put(doc_batch.copy())
    
    # 发送终止信号
    output_queue.put(None)
    print(f"解析完成，共处理{len(md_files)}个文件")


def milvus_writer_process(input_queue: Queue):
    '''进程2：从队列读取并写入Milvus'''
    print('开始写入数据')

    mvs = MilvusVectorSave()
    mvs.create_collection()
    mvs.create_connection()
    total_count = 0
    while True:
        try:
            datas = input_queue.get()  # 阻塞的函数
            if datas is None:
                break
            if isinstance(datas, list):
                mvs.add_documents(datas)
                total_count += len(datas)
                print(f'已经累积写入{total_count}个文档')
        except Exception as e:
            print('写入数据失败!' + str(e))
    print(f'已经累积写入{total_count}个文档')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    # 配置参数
    md_dir = '/root/lanyun-tmp/RAG(Milvus)/project/datasets'
    queue_maxsize = 20
    # 创建进程间的通信队列
    docs_queue = Queue(maxsize=queue_maxsize)

    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, docs_queue)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )
    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    print('---------'*5 + 'end' + '---------'*5)