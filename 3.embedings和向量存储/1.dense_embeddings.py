from langchain_huggingface import HuggingFaceEmbeddings

model_name = "/root/lanyun-tmp/RAG(Milvus)/project/models/BAAI/beg-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
beg_embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

embeddings = beg_embeddings_model.embed_documents(
    [
        'hello World!',
        'Hi there!',
        'oh, hello!',
        "what's your name",
    ]
)

print(len(embeddings), len(embeddings[0]))