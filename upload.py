import chromadb
import os 
import dotenv
from langchain_community.document_loaders import PyPDFLoader
dotenv.load_dotenv()

client = chromadb.PersistentClient(path='db')
collection = client.get_or_create_collection(name="minilm")

import chromadb.utils.embedding_functions as embedding_functions
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.getenv('HUGGINGFACE_API_KEY'),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def db(text, embed, ids):
    collection.add(
    documents=[text],
    embeddings=[embed],
    ids=[str(ids)]
    )
    print(text + " added to database")

def embed(pdf):
    loader = PyPDFLoader(pdf)
    pages = loader.load_and_split()
    for page in pages:
        embeddings = huggingface_ef(page.page_content)
        db(page.page_content, embeddings[0], pages.index(page))

def query_search(input, n=5):
    embedding=huggingface_ef(input)
    
    res=collection.query(
        query_embeddings=[embedding[0]],
        n_results=n,
    )
    # print(res)

    return res['documents'][0]

# embed("eBook-How-to-Build-a-Career-in-AI.pdf")
# print(query_search("who is the author of the book?"))