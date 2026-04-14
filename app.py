import streamlit as st
import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
import pickle, json, time
import matplotlib.pyplot as plt
from model import get_model, get_circuit_diagram

st.set_page_config(page_title="Vahana Rakshaka — QML-IDS", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_models():
    model = get_model()
    model.load_state_dict(torch.load("processed/qml_model.pt", map_location="cpu"))
    model.eval()
    with open("processed/svm.pkl", "rb") as f: svm = pickle.load(f)
    with open("processed/rf.pkl",  "rb") as f: rf  = pickle.load(f)
    return model, svm, rf

@st.cache_data
def load_test_data():
    X = np.load("processed/X_test.npy").astype(np.float32)
    y = np.load("processed/y_test.npy")
    return X, y

@st.cache_data
def load_benchmark():
    with open("processed/benchmark.json") as f:
        return json.load(f)

try:
    model, svm, rf = load_models()
    X_test, y_test = load_test_data()
    benchmark      = load_benchmark()
except Exception as e:
    st.error(
        f"⚠️ Model files not found. Please run `python preprocess.py` "
        f"then `python train.py` before launching the app.\n\nDetails: {e}"
    )
    st.stop()

# ✅ Callback
def load_attack_sample():
    st.session_state["can_id"]  = 2.8
    st.session_state["time_d"]  = 0.1
    st.session_state["dlc"]     = 3.0
    st.session_state["entropy"] = 2.9

# ---------------- HEADER ----------------
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
     padding:2rem;border-radius:12px;margin-bottom:1.5rem;'>
  <h1 style='color:white;margin:0;font-size:2rem;'>🛡️ Vahana Rakshaka</h1>
  <p style='color:#a0aec0;margin:0.3rem 0 0;font-size:1rem;'>
    Hybrid Quantum Machine Learning Intrusion Detection System for CAN Bus Security
  </p>
  <p style='color:#68d391;margin:0.3rem 0 0;font-size:0.85rem;'>
    ⚛️ Powered by Qiskit + PyTorch
  </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Detection",
    "📊 Model Comparison",
    "🔬 Quantum Circuit",
    "🎯 Manual Test"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Real-time CAN Bus Intrusion Detection")
    col1, col2, col3 = st.columns(3)
    total_frames   = col1.empty()
    attacks_found  = col2.empty()
    detection_rate = col3.empty()
    feed_container = st.empty()
    alert_box      = st.empty()

    if st.button("▶ Start Live Simulation", type="primary"):
        history      = []
        attack_count = 0

        for i in range(min(50, len(X_test))):
            x = torch.tensor(X_test[i:i+1], dtype=torch.float32)
            with torch.no_grad():
                prob = model(x).item()

            pred       = int(prob > 0.5)
            true_label = int(y_test[i])
            attack_count += pred
            correct    = (pred == true_label)

            history.append({
                "Frame #": i + 1,
                "Confidence": f"{prob:.2%}",
                "Verdict": "🔴 ATTACK" if pred else "🟢 NORMAL",
                "Correct": "✓" if correct else "✗"
            })

            feed_container.dataframe(pd.DataFrame(history), use_container_width=True, height=300)

            total_frames.metric("Frames", i + 1)
            attacks_found.metric("Attacks", attack_count)
            detection_rate.metric("Rate", f"{attack_count/(i+1):.1%}")

            if pred:
                alert_box.error(f"⚠️ ATTACK — Frame {i+1}")
            else:
                alert_box.success(f"✅ Normal Frame {i+1}")

            time.sleep(0.15)

# ---------------- TAB 2 (FIXED) ----------------
with tab2:
    st.subheader("QML-IDS vs Classical Models")

    try:
        model_names = list(benchmark.keys())
        metrics = ["accuracy", "f1", "false_neg_rate"]

        for metric in metrics:
            vals = [benchmark[m][metric] for m in model_names]

            fig = go.Figure(go.Bar(
                x=model_names,
                y=vals,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside"
            ))

            fig.update_layout(title=metric.upper(), height=300)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading model comparison: {e}")

# ---------------- TAB 3 (FIXED) ----------------
with tab3:
    st.subheader("Quantum Circuit (Qiskit)")

    try:
        qc = get_circuit_diagram()
        fig = qc.draw(output="mpl")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Circuit rendering failed: {e}")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Inject a Custom CAN Frame")

    if "can_id" not in st.session_state: st.session_state["can_id"] = 1.5
    if "time_d" not in st.session_state: st.session_state["time_d"] = 0.3
    if "dlc" not in st.session_state: st.session_state["dlc"] = 1.5
    if "entropy" not in st.session_state: st.session_state["entropy"] = 2.0

    c1, c2 = st.columns(2)

    with c1:
        can_id = st.slider("CAN ID", 0.0, 3.14, key="can_id")
        time_d = st.slider("Time delta", 0.0, 3.14, key="time_d")

    with c2:
        dlc = st.slider("DLC", 0.0, 3.14, key="dlc")
        entropy = st.slider("Entropy", 0.0, 3.14, key="entropy")

    st.caption(f"""
    CAN_ID: {st.session_state['can_id']:.2f} |
    TimeΔ: {st.session_state['time_d']:.2f} |
    DLC: {st.session_state['dlc']:.2f} |
    Entropy: {st.session_state['entropy']:.2f}
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Classify"):
            x = torch.tensor([[can_id, time_d, dlc, entropy]], dtype=torch.float32)
            prob = model(x).item()

            if prob > 0.5:
                st.error(f"ATTACK {prob:.2%}")
            else:
                st.success(f"NORMAL {(1-prob):.2%}")

    with col2:
        st.button("🎲 Load Attack Sample", on_click=load_attack_sample)