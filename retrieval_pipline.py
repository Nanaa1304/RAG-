import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)
query = "Quand Huawei a-t-elle été créée ?"

retriever = db.as_retriever(
      search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3}
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content:\n{doc.page_content}")

    
# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatOpenAI model


model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),  # ou via os.environ / .env
    model="openai/gpt-oss-20b"  # ex: "meta/llama-3.1-70b-instruct"
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)