import gradio as gr
import pickle
import os
from sentence_transformers import SentenceTransformer
# Updated to use the unified multi-modal file processor
from preprocess_multimodal import process_input_audio_or_video
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
    # Check if a file (audio or video) was uploaded
    if uploaded_file is not None:
        # Pass the file path into our multi-modal processor (handles .wav, .mp4, .mov, etc.)
        complaint_text = process_input_audio_or_video(uploaded_file)
    elif text_input and text_input.strip() != "":
        complaint_text = text_input
    else:
        return "System Warning: Please enter text or upload an audio/video file.", "Pending", "Pending", "Pending", "No matching historical contexts found."

    # Prevent pipeline breakdown if file processing returned an error string
    if complaint_text.startswith("Error:"):
        return complaint_text, "Failure", "Failure", "Failure", "Pipeline aborted."

    # Extract Embeddings
    text_features = embedding_model.encode([complaint_text])
    
    # Model Predictions
    officer = officer_model.predict(text_features)[0]
    priority = priority_model.predict(text_features)[0]
    eta = eta_model.predict(text_features)[0]
    
    # Vector DB Similarity Retrieval
    db_matches = find_similar_complaints(complaint_text, top_k=2)
    
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
with gr.Blocks(css=custom_css, title="Grievance AI Engine") as demo:
    
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
                    <span style="background: #334155; color: #E2E8F0; padding: 6px 14px; border-radius: 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.05em; border: 1px solid #475569;">ENGINE_V1.5</span>
                    <span style="background: rgba(16, 185, 129, 0.1); color: #10B981; padding: 6px 14px; border-radius: 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.05em; border: 1px solid rgba(16, 185, 129, 0.3);">MULTIMODAL ENCLAVE</span>
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
                    <span style="padding: 0 16px; font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 0.08em;">OR MULTIMODAL MEDIA FILE</span>
                    <div style="flex-grow: 1; border-top: 1px solid #334155;"></div>
                </div>
                """
            )
            
            # CRITICAL CHANGE: Changed from gr.Audio to gr.File with explicit types allowed to accommodate Audio and Video seamlessly
            file_box = gr.File(
                label="Asynchronous Voice or Video Ingestion", 
                type="filepath",
                file_types=["audio", "video"]
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
            * """
        )

    # Link interactions - inputs updated to target the text box and multi-format file box
    submit_btn.click(
        fn=pipeline_inference, 
        inputs=[text_box, file_box], 
        outputs=[processed_text, out_officer, out_priority, out_eta, out_similarity]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", max_threads=4)