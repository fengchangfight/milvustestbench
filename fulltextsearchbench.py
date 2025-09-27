import os
from typing import List
# from openai import OpenAI
from embeddings import get_embeddings, emb_text


# os.environ["OPENAI_API_KEY"] = "sk-***********"



from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)

uri = "http://localhost:19530"
collection_name = "full_text_demo"
client = MilvusClient(uri=uri, token="root:Milvus")

analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}

schema = MilvusClient.create_schema()
schema.add_field(
    field_name="id",
    datatype=DataType.VARCHAR,
    is_primary=True,
    auto_id=True,
    max_length=100,
)
schema.add_field(
    field_name="content",
    datatype=DataType.VARCHAR,
    max_length=65535,
    analyzer_params=analyzer_params,
    enable_match=True,  # Enable text matching
    enable_analyzer=True,  # Enable text analysis
)
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(
    field_name="dense_vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=384,  # Dimension for all-MiniLM-L6-v2
)
schema.add_field(field_name="metadata", datatype=DataType.JSON)

bm25_function = Function(
    name="bm25",
    function_type=FunctionType.BM25,
    input_field_names=["content"],
    output_field_names="sparse_vector",
)

schema.add_function(bm25_function)

index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
)
index_params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="IP")

if client.has_collection(collection_name):
    client.drop_collection(collection_name)
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,
)
print(f"Collection '{collection_name}' created successfully")

# openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# model_name = "text-embedding-3-small"


# Embedding functions are now imported from embeddings.py

documents = [
    {
        "content": "Milvus is a vector database built for embedding similarity search and AI applications.",
        "metadata": {"source": "documentation", "topic": "introduction"},
    },
    {
        "content": "Full-text search in Milvus allows you to search using keywords and phrases.",
        "metadata": {"source": "tutorial", "topic": "full-text search"},
    },
    {
        "content": "Hybrid search combines the power of sparse BM25 retrieval with dense vector search.",
        "metadata": {"source": "blog", "topic": "hybrid search"},
    },
]

entities = []
texts = [doc["content"] for doc in documents]
embeddings = get_embeddings(texts)

for i, doc in enumerate(documents):
    entities.append(
        {
            "content": doc["content"],
            "dense_vector": embeddings[i],
            "metadata": doc.get("metadata", {}),
        }
    )

client.insert(collection_name, entities)
print(f"Inserted {len(entities)} documents")

