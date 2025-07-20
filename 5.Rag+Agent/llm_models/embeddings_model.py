from langchain_huggingface import HuggingFaceEmbeddings

model_name = "/root/lanyun-tmp/RAG(Milvus)/project/models/BAAI/beg-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
beg_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


if __name__ == '__main__':
    model.query_instruction = "为这个句子生成表示以用于检索相关文章："