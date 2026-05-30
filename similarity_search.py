import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize our local embedding model
# 'all-MiniLM-L6-v2' is lightning fast and runs smoothly on standard CPUs
print("Loading local embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize an offline ChromaDB client
# This saves the data to a local folder called 'chroma_db_storage'
chroma_client = chromadb.PersistentClient(path="chroma_db_storage")

# Create or fetch a collection to store complaints
collection = chroma_client.get_or_create_collection(name="past_complaints")

def index_historical_complaints():
    """Reads dataset.csv, generates embeddings, and saves them to ChromaDB."""
    print("Reading dataset and generating embeddings...")
    df = pd.read_csv("dataset.csv")
    
    # Drop rows without text if any exist
    df = df.dropna(subset=["complaint_text"])
    
    documents = df["complaint_text"].tolist()
    
    # Generate numerical vectors for all complaints at once
    embeddings = embedding_model.encode(documents).tolist()
    
    # Unique IDs for each entry (e.g., "id_0", "id_1", ...)
    ids = [f"id_{i}" for i in range(len(df))]
    
    # Optional metadata so we can trace who the officer was or the priority
    metadatas = df[["assigned_officer", "priority", "eta_days"]].to_dict(orient="records")
    
    # Store everything locally
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully indexed {len(documents)} historical complaints in ChromaDB!")

def find_similar_complaints(new_complaint_text, top_k=2):
    """Finds the most contextually similar past complaints."""
    # Convert incoming new complaint text into a vector
    query_embedding = embedding_model.encode([new_complaint_text]).tolist()
    
    # Query ChromaDB for the closest vector matches using cosine similarity
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return results

if __name__ == "__main__":
    # Index our current dataset
    index_historical_complaints()
    
    # Test a brand new search text that uses different words but has the same MEANING
    test_query = "There's a huge hole in the middle of the road and it's dangerous for motorcycles."
    print(f"\n--- Searching for past complaints similar to: '{test_query}' ---")
    
    matches = find_similar_complaints(test_query, top_k=2)
    
    # Print the results nicely
    for doc, meta, distance in zip(matches['documents'][0], matches['metadatas'][0], matches['distances'][0]):
        print(f"\n[Match Score/Distance: {distance:.4f}]")
        print(f"Complaint: {doc}")
        print(f"Assigned to: {meta['assigned_officer']} | Priority: {meta['priority']}")