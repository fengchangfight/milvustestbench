from pymilvus import MilvusClient
import json
import requests
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
collection_name = "my_rag_collection"

# def emb_text(text):
#     return (
#         openai_client.embeddings.create(input=text, model="text-embedding-3-small")
#         .data[0]
#         .embedding
#     )

def emb_text(text):
    return model.encode(text).tolist()


question = "How is data stored in milvus?"

milvus_client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=3,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)


retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))

context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)


SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""


# openai_client = OpenAI()
# response = openai_client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": USER_PROMPT},
#     ],
# )
# print(response.choices[0].message.content)



def get_llm_response(system_prompt, user_prompt, model="llama2"):
    """Get response from local Ollama model"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': model,
                                   'prompt': f"{system_prompt}\n\n{user_prompt}",
                                   'stream': False
                               })
        return response.json()['response']
    except Exception as e:
        return f"Error: Could not connect to Ollama. Make sure it's running with: ollama serve\nError: {e}"

# Get response from local LLM
response = get_llm_response(SYSTEM_PROMPT, USER_PROMPT)
print(response)
