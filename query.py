from pymilvus import model
from pymilvus import MilvusClient


client = MilvusClient("milvus_demo.db")


# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
embedding_fn = model.DefaultEmbeddingFunction()

query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
# If you don't have the embedding function you can use a fake vector to finish the demo:
# query_vectors = [ [ random.uniform(-1, 1) for _ in range(768) ] ]

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)