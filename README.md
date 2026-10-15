# 🛡️ Real-Time Mobile Click-Fraud & Bot Anomaly Detector

An applied machine learning pipeline engineered to detect automated bot injection attacks, rapid click farm anomalies, and touch telemetry fraud in mobile environments. Uses behavioral telemetry heuristics and cost-sensitive ensemble classification to isolate fraudulent interaction streams with minimal computational overhead.

---

## 📌 Engineering Highlights
- **Telemetry Feature Engineering:** Identifies non-human interaction signatures by extracting touch coordinate variance, touch hold duration, and inter-click latency intervals.
- **Handling Extreme Imbalance:** Employs cost-sensitive balanced tree ensembles designed for high class-imbalance telemetry ($<4\%$ fraud occurrence).
- **Evaluation:** Evaluated on Precision-Recall AUC (PR-AUC) and ROC-AUC rather than misleading vanilla accuracy.
- **Low-Latency Edge Viability:** Lightweight feature extraction requiring negligible CPU and memory resources, suitable for client-side or edge API gateway deployment.

---

## 🏗️ Anomaly Detection Architecture

```text
Touch & Gesture Event Stream (Mobile Client)
                    │
                    ▼
Telemetry Feature Extraction (Hold Duration, Latency, Coordinate Jitter)
                    │
                    ▼
Robust Z-Score Normalization
                    │
                    ▼
Cost-Sensitive Ensemble Scoring (Random Forest / Isolation Classifier)
                    │
                    ▼
Threshold Decision Gate [P(Fraud) > Threshold]
     ├── Flagged (Bot / Automated Click Injection Blocked)
     └── Cleared (Legitimate User Interaction Logged)


git clone [https://github.com/Vishwadhar0000/Mobile-Click-Fraud-Anomaly-Detector.git](https://github.com/Vishwadhar0000/Mobile-Click-Fraud-Anomaly-Detector.git)
cd Mobile-Click-Fraud-Anomaly-Detector
pip install -r requirements.txt
python detect_fraud.py
