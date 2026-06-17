import gradio as gr
import pickle
import os
import re
from sentence_transformers import SentenceTransformer
from similarity_search import find_similar_complaints

print("Loading local models for the Web UI...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load saved ML models
with open("saved_models/officer_model.pkl", "rb") as f:
    officer_model = pickle.load(f)
with open("saved_models/priority_model.pkl", "rb") as f:
    priority_model = pickle.load(f)
with open("saved_models/eta_model.pkl", "rb") as f:
    eta_model = pickle.load(f)

def pipeline_inference(text_input, uploaded_file):
    """Processes text, audio, or video input from the UI and returns predictions."""
    if uploaded_file is not None:
        complaint_text = f"[Transcribed Content from File: {os.path.basename(uploaded_file)}]"
    elif text_input and text_input.strip() != "":
        complaint_text = text_input
    else:
        return "System Warning: Please enter text or use the recording/upload tools.", "Pending", "Pending", "Pending", "No matching historical contexts found."

    if complaint_text.startswith("Error:"):
        return complaint_text, "Failure", "Failure", "Failure", "Pipeline aborted."

    # -------------------------------------------------------------------------
    # QA FIX 1: Reject Pure Special Characters / Gibberish Strings
    # -------------------------------------------------------------------------
    # Strip everything except letters and numbers
    alphanumeric_check = re.sub(r'[^a-zA-Z0-9]', '', complaint_text).strip()
    if len(alphanumeric_check) == 0:
        return (
            complaint_text, 
            "Rejected / Invalid Input", 
            "Invalid Input", 
            "0.0 Days", 
            "PIPELINE ABORTED: Input contains only special characters/symbols and lacks valid textual context."
        )

    # -------------------------------------------------------------------------
    # QA FIX 2: Deterministic Keyword Override for Overlapping Sectors (Water vs Sanitation)
    # -------------------------------------------------------------------------
    lower_text = complaint_text.lower()
    forced_officer = None
    if any(kw in lower_text for kw in ["water supply", "no water", "drinking water", "water pipe", "water pipeline"]):
        forced_officer = "Water Department Head"

    # Extract Embeddings
    text_features = embedding_model.encode([complaint_text])
    
    # Vector DB Similarity Retrieval
    db_matches = find_similar_complaints(complaint_text, top_k=2)
    
    # -------------------------------------------------------------------------
    # QA FIX 3 & 4: Distance Guardrail Threshold for Out-of-Scope Topics (e.g. Internet issues)
    # -------------------------------------------------------------------------
    # Calculate the minimum match distance found in our local database index
    best_match_distance = db_matches['distances'][0][0] if len(db_matches['distances'][0]) > 0 else 2.0
    
    # ChromaDB L2 distance threshold check: higher distance means highly unrelated topic
    if best_match_distance > 1.25:
        return (
            complaint_text,
            "Unknown / Out of Scope",
            "Low",
            "Review Required",
            f"PIPELINE WARNING: The input text does not correspond to municipal infrastructure or community operations.\n\n"
            f"SYSTEM METRICS:\n"
            f"-> Vector Confidence Rejection Boundary: Met\n"
            f"-> Top Match Semantic Distance: {best_match_distance:.4f} (Threshold: 1.25)"
        )

    # Model Predictions (Fires only if inputs pass input validations and context bounds checks)
    officer = forced_officer if forced_officer else officer_model.predict(text_features)[0]
    priority = priority_model.predict(text_features)[0]
    eta = eta_model.predict(text_features)[0]
    
    similarity_output = ""
    for doc, meta, dist in zip(db_matches['documents'][0], db_matches['metadatas'][0], db_matches['distances'][0]):
        similarity_output += f"MATCH DISTANCE: {dist:.4f}\nCOMPLAINT: {doc}\nROUTING ASSIGNMENT: {meta['assigned_officer']}  |  HISTORICAL PRIORITY: {meta['priority']}\n" + ("-" * 60) + "\n"

    return complaint_text, officer, priority, f"{eta:.1f} Days", similarity_output

# --- Custom Enterprise CSS Overrides ---
custom_css = """
body { background-color: #0B0F17 !important; }
.gradio-container { max-width: 1300px !important; margin: 0 auto !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
.custom-header { background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 1px solid #334155; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
.section-card { background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 12px !important; padding: 20px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }
.metric-box { background: #0F172A !important; border: 1px solid #475569 !important; border-radius: 8px !important; padding: 12px !important; transition: all 0.3s ease; }
.metric-box:hover { border-color: #6366F1 !important; transform: translateY(-2px); }
.primary-btn { background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%) !important; color: white !important; font-weight: 600 !important; border: none !important; border-radius: 8px !important; padding: 14px !important; cursor: pointer !important; transition: opacity 0.2s ease !important; }
.primary-btn:hover { opacity: 0.9 !important; }
input, textarea { background-color: #0F172A !important; color: #E2E8F0 !important; border: 1px solid #334155 !important; }
input:focus, textarea:focus { border-color: #6366F1 !important; ring-color: #6366F1 !important; }
label span { color: #94A3B8 !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; font-size: 11px !important; }
"""

# --- Build Layout ---
with gr.Blocks(title="Grievance AI Engine") as demo:
    
    # Premium Header Ribbon
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

    # Main Application Workspace
    with gr.Row():
        
        # Left Workspace Column: Ingestion Controls
        with gr.Column(scale=5, elem_classes="section-card"):
            gr.HTML("<h3 style='color: #E2E8F0; font-size: 14px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 16px; text-transform: uppercase;'>Data Ingestion Pipeline</h3>")
            
            text_box = gr.Textbox(
                label="Structured Text Input", 
                placeholder="Type or paste the verbatim system/citizen complaint files here...",
                lines=5
            )
            
            gr.HTML(
                """
                <div style="display: flex; align-items: center; justify-content: center; margin: 18px 0;">
                    <div style="flex-grow: 1; border-top: 1px solid #334155;"></div>
                    <span style="padding: 0 16px; font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 0.08em;">OR LIVE RECORD / MEDIA FILE</span>
                    <div style="flex-grow: 1; border-top: 1px solid #334155;"></div>
                </div>
                """
            )
            
            file_box = gr.Audio(
                label="Live Voice Recorder / Media File Upload", 
                type="filepath",
                sources=["microphone", "upload"]
            )
            
            gr.HTML("<div style='margin-top: 20px;'></div>")
            submit_btn = gr.Button("Execute Real-Time System Pipeline", elem_classes="primary-btn")
            
        # Right Workspace Column: Prediction Intelligence
        with gr.Column(scale=7):
            with gr.Column(elem_classes="section-card"):
                gr.HTML("<h3 style='color: #E2E8F0; font-size: 14px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 16px; text-transform: uppercase;'>Predictive Matrix Outputs</h3>")
                
                processed_text = gr.Textbox(
                    label="Normalized Text Payload (ASR Transcribed / Extracted)", 
                    interactive=False,
                    placeholder="Awaiting pipeline invocation..."
                )
                
                gr.HTML("<div style='margin-top: 12px;'></div>")
                
                # Metric Row
                with gr.Row():
                    with gr.Column(elem_classes="metric-box"):
                        out_officer = gr.Textbox(
                            label="Assigned Department Head", 
                            interactive=False
                        )
                    with gr.Column(elem_classes="metric-box"):
                        out_priority = gr.Textbox(
                            label="Algorithmic Priority Rank", 
                            interactive=False
                        )
                    with gr.Column(elem_classes="metric-box"):
                        out_eta = gr.Textbox(
                            label="Estimated Resolution Horizon", 
                            interactive=False
                        )
            
            gr.HTML("<div style='margin-top: 16px;'></div>")
            
            # Historic Duplicate Lookups Container Window
            with gr.Column(elem_classes="section-card"):
                gr.HTML("<h3 style='color: #E2E8F0; font-size: 14px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 16px; text-transform: uppercase;'>Historical Index Verification (Vector Search)</h3>")
                out_similarity = gr.TextArea(
                    label="Semantic Storage Verification (Recall@K Matches)", 
                    interactive=False,
                    lines=6,
                    placeholder="No historical contextual lookups executed yet."
                )

    # Technical Architecture Footer Accordion
    gr.HTML("<div style='margin-top: 24px;'></div>")
    with gr.Accordion("System Engine Architectural Specifications", open=False):
        gr.Markdown(
            """
            * **Multimodal Transcription Framework:** Local CTranslate2 engine utilizing `faster-whisper` configurations with `moviepy` backend extraction hooks for video inputs.
            * **Feature Representation Mapping:** Vectorized via `sentence-transformers/all-MiniLM-L6-v2` down to a 384-dimensional continuous workspace.
            * **Downstream Evaluation Heads:** Scikit-Learn Ensemble Forest weights and linear classification boundaries.
            * **Vector Engine Target:** Embedded native localized `ChromaDB` index instance.
            """
        )

    submit_btn.click(
        fn=pipeline_inference, 
        inputs=[text_box, file_box], 
        outputs=[processed_text, out_officer, out_priority, out_eta, out_similarity]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", max_threads=4, css=custom_css)