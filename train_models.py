import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error

# 1. Setup workspace
os.makedirs("saved_models", exist_ok=True)

print("Loading dataset and text embedding extractor...")
df = pd.read_csv("dataset.csv")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Transforming complaint texts into dense vector features...")
X = embedding_model.encode(df["complaint_text"].tolist())

# Extract prediction target arrays
y_officer = df["assigned_officer"]
y_priority = df["priority"]
y_eta = df["eta"]

# --- Train Task 1: Officer Routing ---
X_train, X_test, y_train, y_test = train_test_split(X, y_officer, test_size=0.25, random_state=42)
officer_model = RandomForestClassifier(n_estimators=100, random_state=42)
officer_model.fit(X_train, y_train)
with open("saved_models/officer_model.pkl", "wb") as f:
    pickle.dump(officer_model, f)

# --- Train Task 2: Priority Classifier ---
X_train, X_test, y_train, y_test = train_test_split(X, y_priority, test_size=0.25, random_state=42)
priority_model = RandomForestClassifier(n_estimators=100, random_state=42)
priority_model.fit(X_train, y_train)
with open("saved_models/priority_model.pkl", "wb") as f:
    pickle.dump(priority_model, f)

# --- Train Task 3: ETA Regressor ---
X_train, X_test, y_train, y_test = train_test_split(X, y_eta, test_size=0.25, random_state=42)
eta_model = RandomForestRegressor(n_estimators=100, random_state=42)
eta_model.fit(X_train, y_train)
with open("saved_models/eta_model.pkl", "wb") as f:
    pickle.dump(eta_model, f)

print("🚀 Core weights optimized! All models retrained and saved inside /saved_models/ folder successfully.")