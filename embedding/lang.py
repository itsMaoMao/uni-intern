from langchain.document_loaders import TextLoader
from langchain.embeddings import ModelScopeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os
os.environ['MODELSCOPE_NO_NETWORK']='1'
# import chardet

import chromadb
# chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(path="./database")

# 读取原始文档
raw_documents_sanguo = TextLoader('./三国演义_fenci.txt', encoding='utf-8').load()
raw_documents_xiyou = TextLoader('./西游记_fenci.txt', encoding='utf-8').load()

# 分割文档
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
documents_sanguo = text_splitter.split_documents(raw_documents_sanguo)
documents_xiyou = text_splitter.split_documents(raw_documents_xiyou)
documents = documents_sanguo + documents_xiyou
print("documents nums:", documents.__len__())

# 生成向量（embedding）
model_id = "./model/nlp_corom_sentence-embedding_chinese-base"
embedding_model = ModelScopeEmbeddings(model_id=model_id)

db = Chroma.from_documents(documents,   
                           collection_name="xiyouAndSanguo",  # 可选，collection 名称
                           client=chroma_client,  # 可选，已有的 chromadb.Client())
                            embedding = embedding_model   )       
# 检索
query = "美猴王是谁？"
docs = db.similarity_search(query, k=5)

# 打印结果
for doc in docs:
    print("===")
    print("metadata:", doc.metadata)
    # print("page_content:", doc.page_content)