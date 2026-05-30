# Municipal Grievance Intelligence & Auto-Routing System

A production-ready, 100% offline, model-driven artificial intelligence pipeline architected to ingest multi-format citizen grievances via raw text payloads or acoustic audio waveforms. The system automatically performs downstream administrative routing, algorithmic hazard priority profiling, regression timeline predictions (ETA), and persistent vector similarity indexing.

---

## 🏛️ 1. Machine Learning Problem Framing

This platform approaches the handling of unstructured citizen inputs entirely from an objective, data-driven perspective, completely bypassing rule-based keywords or manual mapping logic:

* **Task 1 (Department Routing):** Formulated as an **Embedding-Driven Multi-class Classification** problem. Unstructured textual context vectors are projected onto discrete administrative bureau domains (`Water`, `Sanitation`, `Electrical`, `Emergency`).
* **Task 2 (Priority Allocation):** Formulated as a **Categorical Ordinal Classification** task, bucketing instances mathematically into Low, Medium, or High tracking states based on semantic hazard severity weights.
* **Task 3 (Resolution ETA):** Formulated as a **Continuous Numerical Regression** problem, mapping high-dimensional text embeddings directly to a linear temporal space representing days until resolution.
* **Task 4 (Duplicate Case Discovery):** Formulated as an **Unsupervised Semantic Search Vector Space Inference** problem, identifying historical context clusters using geometric cosine distances.

---

## ⚙️ 2. Architectural Data Pipeline Flow

```text
[Audio Input (.wav)] ──> Whisper ASR Engine ──┐
                                             ├──> [Normalized Text] ──> all-MiniLM-L6 Embedding (384-D)
[Text Input (Raw)]   ────────────────────────┘                                │
                                                                              ├─> RandomForestClassifier ─> [Dept Head]
                                                                              ├─> RandomForestClassifier ─> [Priority]
                                                                              ├─> RandomForestRegressor  ─> [Resolution ETA]
                                                                              └─> ChromaDB Index (L2)    ─> [Historical Matches]