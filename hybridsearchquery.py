
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from embeddings import get_embeddings, get_llm_response_single


uri = "http://localhost:19530"
collection_name = "full_text_demo"
client = MilvusClient(uri=uri, token="root:Milvus")

query = "full-text search keywords"

results = client.search(
    collection_name=collection_name,
    data=[query],
    anns_field="sparse_vector",
    limit=5,
    output_fields=["content", "metadata"],
)
sparse_results = results[0]

print("\nSparse Search (Full-text search):")
for i, result in enumerate(sparse_results):
    print(
        f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}"
    )

query = "How does Milvus help with similarity search?"

query_embedding = get_embeddings([query])[0]

results = client.search(
    collection_name=collection_name,
    data=[query_embedding],
    anns_field="dense_vector",
    limit=5,
    output_fields=["content", "metadata"],
)
dense_results = results[0]

print("\nDense Search (Semantic):")
for i, result in enumerate(dense_results):
    print(
        f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}"
    )




query = "what is hybrid search"

query_embedding = get_embeddings([query])[0]

sparse_search_params = {"metric_type": "BM25"}
sparse_request = AnnSearchRequest(
    [query], "sparse_vector", sparse_search_params, limit=5
)

dense_search_params = {"metric_type": "IP"}
dense_request = AnnSearchRequest(
    [query_embedding], "dense_vector", dense_search_params, limit=5
)

results = client.hybrid_search(
    collection_name,
    [sparse_request, dense_request],
    ranker=RRFRanker(),  # Reciprocal Rank Fusion for combining results
    limit=5,
    output_fields=["content", "metadata"],
)
hybrid_results = results[0]

print("\nHybrid Search (Combined):")
for i, result in enumerate(hybrid_results):
    print(
        f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}"
    )


context = "\n\n".join([doc["entity"]["content"] for doc in hybrid_results])

prompt = f"""Answer the following question based on the provided context. 
If the context doesn't contain relevant information, just say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""


get_llm_response_single(prompt)

# Note: OpenAI client is not imported in this file
# Uncomment and import if you want to use OpenAI for responses
# from openai import OpenAI
# openai_client = OpenAI()
# response = openai_client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that answers questions based on the provided context.",
#         },
#         {"role": "user", "content": prompt},
#     ],
# )
# print(response.choices[0].message.content)


