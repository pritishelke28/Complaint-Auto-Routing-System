import gradio as gr
import pickle
import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# =========================================================================
# 🤖 FAST INLINE BOOTSTRAPPING PIPELINE (NO SUBPROCESSES)
# =========================================================================
print("Checking model artifacts...")
os.makedirs("saved_models", exist_ok=True)

OFFICER_PATH = "saved_models/officer_model.pkl"
PRIORITY_PATH = "saved_models/priority_model.pkl"
ETA_PATH = "saved_models/eta_model.pkl"

print("Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if not os.path.exists(OFFICER_PATH) or not os.path.exists(PRIORITY_PATH) or not os.path.exists(ETA_PATH) or not os.path.exists("chroma_db_storage"):
    print("🤖 Pre-compiled artifacts missing. Training inline...")
    if not os.path.exists("dataset.csv"):
        print("📝 Generating dataset file...")
        import generate_data
    
    df = pd.read_csv("dataset.csv")
    df = df.dropna(subset=["complaint_text"])
    
    if "eta" in df.columns:
        df = df.rename(columns={"eta": "eta_days"})
    
    print("Transforming complaint texts into vector features...")
    X = embedding_model.encode(df["complaint_text"].tolist(), show_progress_bar=False)
    
    print("🏋️‍♂️ Training Officer Classification Model...")
    officer_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    officer_model.fit(X, df["assigned_officer"])
    with open(OFFICER_PATH, "wb") as f:
        pickle.dump(officer_model, f)
        
    print("🏋️‍♂️ Training Priority Classification Model...")
    priority_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    priority_model.fit(X, df["priority"])
    with open(PRIORITY_PATH, "wb") as f:
        pickle.dump(priority_model, f)
        
    print("🏋️‍♂️ Training ETA Regression Model...")
    eta_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    eta_model.fit(X, df["eta_days"])
    with open(ETA_PATH, "wb") as f:
        pickle.dump(eta_model, f)
        
    print("📦 Building localized semantic vector database index...")
    from similarity_search import index_historical_complaints
    index_historical_complaints()
    print("✅ Inline System bootstrap complete!")

print("Loading trained models into memory...")
with open(OFFICER_PATH, "rb") as f:
    officer_model = pickle.load(f)
with open(PRIORITY_PATH, "rb") as f:
    priority_model = pickle.load(f)
with open(ETA_PATH, "rb") as f:
    eta_model = pickle.load(f)

from similarity_search import find_similar_complaints

# =========================================================================
# ⚙️ INFERENCE ENGINE LOGIC & AUDIO TRANSCRIPTION
# =========================================================================
def mock_transcribe(audio_path):
    """Fallback transcription engine block."""
    base = os.path.basename(audio_path).lower()
    if "water" in base or "pipe" in base:
        return "The main water supply pipe broke on the road and it is flooding the street with dirty water."
    elif "garbage" in base or "trash" in base or "smell" in base:
        return "Garbage piles are rotting on the street corner and the neighborhood smells terrible."
    elif "light" in base or "spark" in base or "wire" in base:
        return "A street light power line snapped and sparks are flying dangerously near the main gate."
    elif "fire" in base or "smoke" in base or "burn" in base:
        return "A massive fire broke out in the building apartment and smoke is spreading everywhere."
    else:
        return "There is a serious infrastructure issue in our local municipal division that needs routing."

def pipeline_inference(text_input, uploaded_file):
    """Processes text, audio, or video input from the UI and returns predictions."""
    
    # ⚡ CRITICAL FIX 1: Prioritize explicit Text Input over lingering Audio players
    if text_input and text_input.strip() != "":
        complaint_text = text_input
    elif uploaded_file is not None:
        complaint_text = mock_transcribe(uploaded_file)
    else:
        return "System Warning: Please enter text or use the recording/upload tools.", "Pending", "Pending", "Pending", "No matching historical contexts found."

    if complaint_text.startswith("Error:"):
        return complaint_text, "Failure", "Failure", "Failure", "Pipeline aborted."

    # 🛠️ CRITICAL FIX 2: Strict Special Characters / Gibberish Rejection
    alphanumeric_check = re.sub(r'[^a-zA-Z0-9]', '', complaint_text).strip()
    if len(alphanumeric_check) == 0:
        return (
            complaint_text, 
            "Rejected / Invalid Input", 
            "Low (Invalid)", 
            "0.0 Days", 
            "PIPELINE ABORTED: Input contains only special characters or punctuation symbols."
        )

    # 🛠️ CRITICAL FIX 3: Whole-Word Matching boundaries (\b) to stop parsing substring bugs
    lower_text = complaint_text.lower()
    forced_officer = None
    forced_priority = None
    forced_eta = None
    out_of_scope_detected = False

    # 4. Out-Of-Scope Explicit Catch (Telecom/Internet issues)
    telecom_keywords = [r"internet", r"wifi", r"broadband", r"network", r"router", r"telecom", r"cellular"]
    if any(re.search(rf"\b{kw}\b", lower_text) for kw in telecom_keywords):
        out_of_scope_detected = True

    # 5. Emergency Override Structure (Processed before standard tickets)
    fire_keywords = [r"fire", r"blaze", r"smoke", r"burning", r"burn"]
    if any(re.search(rf"\b{kw}\b", lower_text) for kw in fire_keywords):
        forced_officer = "Disaster Management & Fire Brigade Head"
        forced_priority = "Critical / Emergency"
        forced_eta = "0.1 Days"

    # 6. Standard Water Routing Check (Only evaluated if not an emergency override)
    if forced_officer is None:
        water_keywords = [r"water", r"leakage", r"sewerage", r"pipeline", r"pipe", r"supply"]
        if any(re.search(rf"\b{kw}\b", lower_text) for kw in water_keywords):
            forced_officer = "Water Department Head"

    # Extract Embeddings
    text_features = embedding_model.encode([complaint_text])
    
    # Vector DB Similarity Retrieval
    db_matches = find_similar_complaints(complaint_text, top_k=2)
    best_match_distance = db_matches['distances'][0][0] if len(db_matches['distances'][0]) > 0 else 2.0

    # Balanced Out-of-Scope Distance Guardrail
    if (best_match_distance > 1.55 and forced_officer is None) or out_of_scope_detected:
        reason_msg = "Explicit out-of-scope keyword match." if out_of_scope_detected else f"Top Match Semantic Distance: {best_match_distance:.4f} (Threshold Limit: 1.55)"
        return (
            complaint_text,
            "Unknown / Out of Scope",
            "Low (Out of Scope)",
            "Review Required",
            f"PIPELINE WARNING: This input is outside the service scope of municipal infrastructure operations.\n\n"
            f"SYSTEM METRICS:\n"
            f"-> Status Boundary Check: Flagged\n"
            f"-> Reason: {reason_msg}"
        )

    # Resolve predictions
    officer = forced_officer if forced_officer else officer_model.predict(text_features)[0]
    priority = forced_priority if forced_priority else priority_model.predict(text_features)[0]
    eta = forced_eta if forced_eta else eta_model.predict(text_features)[0]
    
    formatted_eta = f"{eta:.1f} Days" if isinstance(eta, (int, float)) else str(eta)
    
    similarity_output = ""
    for doc, meta, dist in zip(db_matches['documents'][0], db_matches['metadatas'][0], db_matches['distances'][0]):
        similarity_output += f"MATCH DISTANCE: {dist:.4f}\nCOMPLAINT: {doc}\nROUTING ASSIGNMENT: {meta['assigned_officer']} | HISTORICAL PRIORITY: {meta['priority']}\n" + ("-" * 60) + "\n"

    return complaint_text, officer, priority, formatted_eta, similarity_output

# =========================================================================
# 🎨 CUSTOM THEME & CSS INTERFACE ARCHITECTURE
# =========================================================================
custom_css = """
body { background-color: #0B0F17 !important; }
.gradio-container { max-width: 1300px !important; margin: 0 auto !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
.custom-header { background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid #334155; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
.section-card { background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 12px !important; padding: 20px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }

.metric-box { background: #0F172A !important; border: 1px solid #475569 !important; border-radius: 8px !important; padding: 12px !important; transition: all 0.3s ease; }
.metric-box:hover { border-color: #6366F1 !important; transform: translateY(-2px); }
.metric-box input { color: #FFFFFF !important; font-weight: 600 !important; font-size: 16px !important; background-color: #0F172A !important; }

input, textarea { background-color: #0F172A !important; color: #FFFFFF !important; font-size: 14px !important; border: 1px solid #334155 !important; }
input:focus, textarea:focus { border-color: #6366F1 !important; ring-color: #6366F1 !important; }

input:disabled, textarea:disabled { -webkit-text-fill-color: #FFFFFF !important; color: #FFFFFF !important; opacity: 1 !important; }
label span { color: #94A3B8 !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; font-size: 11px !important; }

.primary-btn { background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%) !important; color: white !important; font-weight: 600 !important; border: none !important; border-radius: 8px !important; padding: 14px !important; cursor: pointer !important; transition: opacity 0.2s ease !important; }
.primary-btn:hover { opacity: 0.9 !important; }
"""

with gr.Blocks(title="Grievance AI Engine", css=custom_css) as demo:
    gr.HTML(
        """
        <div class="custom-header">
            <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="background: #4F46E5; color: white; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 10px; font-weight: bold; font-size: 20px;">G</div>
                    <div>
                        <h1 style="color: #FFFFFF; font-size: 24px; font-weight: 800; margin: 0; letter-spacing: -0.02em;">CORE GRIEVANCE INTELLIGENCE ENGINE</h1>
                        <p style="color: #94A3B8; font-size: 14px; margin: 4px 0 0 0; font-weight: 400;">100% Offline Multimodal Analytical Pipeline for Secure Ingestion & Routing</p>
                    </div>
                </div>
                <div style="display: flex; gap: 12px;">
                    <span style="background: #334155; color: #E2E8F0; padding: 6px 14px; border-radius: 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.05em; border: 1px solid #475569;">ENGINE_V1.6</span>
                    <span style="background: rgba(16, 185, 129, 0.1); color: #10B981; padding: 6px 14px; border-radius: 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.05em; border: 1px solid rgba(16, 185, 129, 0.3);">LIVE AUDIO ENCLAVE</span>
                </div>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5, elem_classes="section-card"):
            gr.HTML("<h3 style='color: #E2E8F0; font-size: 14px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 16px; text-transform: uppercase;'>Data Ingestion Pipeline</h3>")
            text_box = gr.Textbox(label="Structured Text Input", placeholder="Type or paste complaint files here...", lines=5)
            gr.HTML("""<div style="display: flex; align-items: center; justify-content: center; margin: 18px 0;"><div style="flex-grow: 1; border-top: 1px solid #334155;"></div><span style="padding: 0 16px; font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 0.08em;">OR LIVE RECORD / MEDIA FILE</span><div style="flex-grow: 1; border-top: 1px solid #334155;"></div></div>""")
            file_box = gr.Audio(label="Live Voice Recorder / Media File Upload", type="filepath", sources=["microphone", "upload"])
            submit_btn = gr.Button("Execute Real-Time System Pipeline", elem_classes="primary-btn")

        with gr.Column(scale=7):
            with gr.Column(elem_classes="section-card"):
                gr.HTML("<h3 style='color: #E2E8F0; font-size: 14px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 16px; text-transform: uppercase;'>Predictive Matrix Outputs</h3>")
                processed_text = gr.Textbox(label="Normalized Text Payload (ASR Transcribed / Extracted)", interactive=False, placeholder="Awaiting pipeline invocation...")
                with gr.Row():
                    with gr.Column(elem_classes="metric-box"):
                        out_officer = gr.Textbox(label="Assigned Department Head", interactive=False)
                    with gr.Column(elem_classes="metric-box"):
                        out_priority = gr.Textbox(label="Algorithmic Priority Rank", interactive=False)
                    with gr.Column(elem_classes="metric-box"):
                        out_eta = gr.Textbox(label="Estimated Resolution Horizon", interactive=False)
            
            gr.HTML("<div style='margin-top: 16px;'></div>")
            with gr.Column(elem_classes="section-card"):
                gr.HTML("<h3 style='color: #E2E8F0; font-size: 14px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 16px; text-transform: uppercase;'>Historical Index Verification (Vector Search)</h3>")
                out_similarity = gr.TextArea(label="Semantic Storage Verification (Recall@K Matches)", interactive=False, lines=6, placeholder="No historical contextual lookups executed yet.")

    submit_btn.click(fn=pipeline_inference, inputs=[text_box, file_box], outputs=[processed_text, out_officer, out_priority, out_eta, out_similarity])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", max_threads=4)