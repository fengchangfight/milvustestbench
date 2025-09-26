import os
from glob import glob
from openai import OpenAI
from pymilvus import MilvusClient
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


# os.environ["OPENAI_API_KEY"] = "sk-***********"


text_lines = []

for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")


# openai_client = OpenAI()

def emb_text(text):
    return model.encode(text).tolist()

# def emb_text(text):
#     return (
#         openai_client.embeddings.create(input=text, model="text-embedding-3-small")
#         .data[0]
#         .embedding
#     )

test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])



milvus_client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Bounded",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
)

data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

milvus_client.insert(collection_name=collection_name, data=data)
