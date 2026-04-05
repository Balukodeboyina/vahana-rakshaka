import streamlit as st
import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
import pickle, json, time
import pennylane as qml
import matplotlib.pyplot as plt
from model import get_model, quantum_circuit, n_qubits, n_layers

st.set_page_config(page_title="Vahana Rakshaka — QML-IDS",
                   page_icon="🛡️", layout="wide")

# ---- Load models with error handling ----
@st.cache_resource
def load_models():
    model = get_model()
    model.load_state_dict(torch.load("processed/qml_model.pt",
                                      map_location="cpu"))
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

# --- Safe startup: show friendly error if training hasn't been run yet ---
try:
    model, svm, rf = load_models()
    X_test, y_test = load_test_data()
    benchmark      = load_benchmark()
    models_loaded  = True
except Exception as e:
    models_loaded = False
    st.error(
        f"⚠️ Model files not found. Please run `python preprocess.py` "
        f"then `python train.py` before launching the app.\n\nDetails: {e}"
    )
    st.stop()

# ---- Header ----
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
     padding:2rem;border-radius:12px;margin-bottom:1.5rem;'>
  <h1 style='color:white;margin:0;font-size:2rem;'>🛡️ Vahana Rakshaka</h1>
  <p style='color:#a0aec0;margin:0.3rem 0 0;font-size:1rem;'>
    Hybrid Quantum Machine Learning Intrusion Detection System for CAN Bus Security
  </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Detection",
    "📊 Model Comparison",
    "🔬 Quantum Circuit",
    "🎯 Manual Test"
])

# ========== TAB 1: LIVE DETECTION ==========
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
            x = torch.tensor(X_test[i:i+1], dtype=torch.float32)  # FIX: explicit dtype
            with torch.no_grad():
                prob = model(x).item()
            pred       = int(prob > 0.5)
            true_label = int(y_test[i])                            # FIX: renamed from 'true'
            attack_count += pred
            correct    = (pred == true_label)
            history.append({
                "Frame #":       i + 1,
                "CAN ID (norm)": f"{X_test[i][0]:.3f}",
                "Time Δ":        f"{X_test[i][1]:.3f}",
                "DLC":           f"{X_test[i][2]:.3f}",
                "Entropy":       f"{X_test[i][3]:.3f}",
                "Confidence":    f"{prob:.2%}",
                "Verdict":       "🔴 ATTACK" if pred else "🟢 NORMAL",
                "Correct":       "✓" if correct else "✗"
            })
            df = pd.DataFrame(history)
            feed_container.dataframe(df, use_container_width=True, height=300)
            total_frames.metric("Frames Processed", i + 1)
            attacks_found.metric("Attacks Detected", attack_count)
            detection_rate.metric("Detection Rate", f"{attack_count / (i + 1):.1%}")
            if pred:
                alert_box.error(
                    f"⚠️ INTRUSION DETECTED — Frame {i+1} | Confidence: {prob:.2%}"
                )
            else:
                alert_box.success(f"✅ Frame {i+1} — Normal CAN traffic")
            time.sleep(0.15)

# ========== TAB 2: BENCHMARK ==========
with tab2:
    st.subheader("QML-IDS vs Classical Models")
    model_names = list(benchmark.keys())
    metrics     = ["accuracy", "f1", "false_neg_rate"]
    labels      = ["Accuracy", "F1 Score", "False Negative Rate (lower=better)"]
    colors      = ["#7F77DD", "#888780", "#5DCAA5"]

    for metric, label, color in zip(metrics, labels, colors):
        vals = [benchmark[m][metric] for m in model_names]
        fig  = go.Figure(go.Bar(
            x=model_names,
            y=vals,
            marker_color=[color if m == "QML-IDS (Ours)" else "#d3d1c7"
                          for m in model_names],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside"
        ))
        fig.update_layout(
            title=label, height=300,
            yaxis_range=[0, max(vals) * 1.2],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Results Summary Table")
    rows = []
    for name, r in benchmark.items():
        rows.append({
            "Model":               name,
            "Accuracy":            f"{r['accuracy']:.4f}",
            "F1 Score":            f"{r['f1']:.4f}",
            "False Negative Rate": f"{r['false_neg_rate']:.4f}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ========== TAB 3: QUANTUM CIRCUIT ==========
with tab3:
    st.subheader("The Variational Quantum Circuit (VQC)")
    st.markdown("""
    This is the quantum core of Vahana Rakshaka. The circuit encodes CAN bus features
    as rotation angles on 4 qubits, then applies entangling layers where the vehicle's
    attack signature becomes a quantum state that cannot be efficiently represented classically.
    """)

    # FIX: explicit float32 tensors for circuit drawing
    weights      = np.random.uniform(0, np.pi, (n_layers, n_qubits, 3))
    sample_input = np.array([1.2, 0.5, 2.1, 0.8])
    try:
        fig_qnode = qml.draw_mpl(quantum_circuit, decimals=2)(
            torch.tensor(sample_input, dtype=torch.float32),
            torch.tensor(weights,      dtype=torch.float32)
        )
        st.pyplot(fig_qnode[0])
        plt.close()
    except Exception as e:
        st.info(f"Circuit diagram could not render: {e}")

    st.markdown("---")
    st.subheader("How it works — step by step")

    with st.expander("Step 1: Feature encoding via AngleEmbedding"):
        st.markdown("""
        Each of the 4 CAN bus features (CAN ID, time delta, DLC, entropy) is encoded
        as a rotation angle on one qubit using an Rx gate:

        `Rx(feature_value) |0⟩`

        This maps classical data into the quantum Hilbert space.
        """)

    with st.expander("Step 2: StronglyEntanglingLayers (the trainable part)"):
        st.markdown("""
        3 layers of parameterized rotation + CNOT entanglement gates.
        Total trainable parameters: 3 layers × 4 qubits × 3 rotations = **36 parameters**.
        These weights are learned during training via gradient descent on the quantum circuit.
        """)

    with st.expander("Step 3: Pauli-Z measurement → attack probability"):
        st.markdown("""
        The expectation value ⟨Z₀⟩ on qubit 0 returns a value in [-1, +1].
        This is passed through a sigmoid function to give a probability [0, 1].
        Above 0.5 = ATTACK. Below 0.5 = NORMAL.
        """)

# ========== TAB 4: MANUAL TEST ==========
with tab4:
    st.subheader("Inject a Custom CAN Frame")
    st.markdown("Simulate what a hacker might send — or test a normal frame.")

    # FIX: use session_state so 'Load Attack Sample' button actually changes slider values
    if "can_id" not in st.session_state: st.session_state["can_id"] = 1.5
    if "time_d" not in st.session_state: st.session_state["time_d"] = 0.3
    if "dlc"    not in st.session_state: st.session_state["dlc"]    = 1.5
    if "entropy" not in st.session_state: st.session_state["entropy"] = 2.0

    c1, c2 = st.columns(2)
    with c1:
        can_id = st.slider("CAN ID (normalized 0–π)",       0.0, 3.14,
                           st.session_state["can_id"],  step=0.01, key="can_id")
        time_d = st.slider("Time delta (ms, normalized)",   0.0, 3.14,
                           st.session_state["time_d"],  step=0.01, key="time_d")
    with c2:
        dlc    = st.slider("DLC value (normalized)",        0.0, 3.14,
                           st.session_state["dlc"],     step=0.01, key="dlc")
        entropy = st.slider("Data entropy (normalized)",    0.0, 3.14,
                            st.session_state["entropy"], step=0.01, key="entropy")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🔍 Classify this frame"):
            x = torch.tensor([[can_id, time_d, dlc, entropy]],
                             dtype=torch.float32)               # FIX: explicit dtype
            with torch.no_grad():
                prob = model(x).item()
            if prob > 0.5:
                st.error(f"⚠️ ATTACK DETECTED — Confidence: {prob:.2%}")
            else:
                st.success(f"✅ NORMAL TRAFFIC — Confidence: {(1 - prob):.2%}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Attack Probability (%)"},
                gauge={
                    "axis":      {"range": [0, 100]},
                    "bar":       {"color": "#E24B4A"},
                    "steps":     [{"range": [0, 50],   "color": "#EAF3DE"},
                                  {"range": [50, 100], "color": "#FCEBEB"}],
                    "threshold": {"line": {"color": "red", "width": 4},
                                  "thickness": 0.75, "value": 50}
                }
            ))
            fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with col_btn2:
        # FIX: button now loads realistic attack values instead of doing nothing
        if st.button("🎲 Load Attack Sample"):
            st.session_state["can_id"]  = 2.8
            st.session_state["time_d"]  = 0.1
            st.session_state["dlc"]     = 3.0
            st.session_state["entropy"] = 2.9
            st.rerun()