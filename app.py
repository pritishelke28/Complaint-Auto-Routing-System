import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from preprocess_multimodal import transcribe_audio
from similarity_search import find_similar_complaints

# 1. Load the local embedding model
print("Initializing pipeline and loading local models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load the trained machine learning models
try:
    with open("saved_models/officer_model.pkl", "rb") as f:
        officer_model = pickle.load(f)
    with open("saved_models/priority_model.pkl", "rb") as f:
        priority_model = pickle.load(f)
    with open("saved_models/eta_model.pkl", "rb") as f:
        eta_model = pickle.load(f)
except FileNotFoundError:
    print("\n[ERROR] Trained models not found! Please run 'python train_models.py' first.")
    exit()

def process_new_complaint(input_data, is_file=False):
    """
    End-to-End Pipeline:
    Takes an input (Text or Audio/Video Path), handles multimodal ingestion, 
    extracts features, makes predictions using ML models, and fetches similar past cases.
    """
    print("\n" + "="*50)
    print(" PROCESSING INCOMING COMPLAINT ".center(50, "="))
    print("="*50)
    
    # Check if input is a file (Audio or Video) or direct Text
    if is_file:
        print(f"File input detected. Sending to local Whisper engine...")
        complaint_text = transcribe_audio(input_data)
        print(f"Transcribed Text: \"{complaint_text}\"")
    else:
        complaint_text = input_data
        print(f"Direct text input detected: \"{complaint_text}\"")
        
    # Generate vector embeddings for the text
    text_features = embedding_model.encode([complaint_text])
    
    # Run Inference using our trained ML models (No rule-based logic!)
    predicted_officer = officer_model.predict(text_features)[0]
    predicted_priority = priority_model.predict(text_features)[0]
    predicted_eta = eta_model.predict(text_features)[0]
    
    # Query vector database for similar past cases (Recall@K verification)
    db_matches = find_similar_complaints(complaint_text, top_k=2)
    
    # Output the structured prediction report
    print("\n" + "-"*15 + " ML INFERENCE ENGINE RESULTS " + "-"*15)
    print(f" Assigned Officer : {predicted_officer}")
    print(f" Predicted Priority: {predicted_priority}")
    print(f" Estimated ETA     : {predicted_eta:.1f} days")
    print("-"*59)
    
    print("\n" + "-"*16 + " SEMANTIC RETRIEVAL MATCHES " + "-"*16)
    for doc, meta, distance in zip(db_matches['documents'][0], db_matches['metadatas'][0], db_matches['distances'][0]):
        # Distance calculation helps verify model-driven retrieval confidence
        print(f" -> [Match Dist: {distance:.4f}] {doc}")
        print(f"    Resolved by: {meta['assigned_officer']} | Past Priority: {meta['priority']}\n")
    print("="*50)

if __name__ == "__main__":
    # Test 1: Let's process a direct text complaint (Simulating an online text submission)
    text_complaint = "The street garbage collection bins are breaking down and spilling all over the road. Foul smell everywhere."
    process_new_complaint(text_complaint, is_file=False)
    
    # Test 2: Let's process our real local audio file we created earlier!
    audio_file_path = "mock_uploads/complaint_02.wav"
    process_new_complaint(audio_file_path, is_file=True)