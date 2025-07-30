from langchain.document_loaders import TextLoader
from langchain.embeddings import ModelScopeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
# import chardet

import chromadb
# chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(path="./database")

model_id = "./model/nlp_corom_sentence-embedding_chinese-base"
embedding_model = ModelScopeEmbeddings(model_id=model_id)

db = Chroma(  
            persist_directory="./database",
            collection_name="xiyouAndSanguo",  # 可选，collection 名称
            client=chroma_client,  # 可选，已有的 chromadb.Client())
            embedding_function = embedding_model   )       
# 检索
query = "美猴王是谁？"
docs = db.similarity_search(query, k=5)

# 打印结果
for doc in docs:
    print("===")
    print("metadata:", doc.metadata)
    # print("page_content:", doc.page_content)