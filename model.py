import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn

# --- Quantum device: 4 qubits, runs on your CPU (no real quantum hardware needed) ---
n_qubits = 4
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Step 1: Encode the 4 features as rotation angles on each qubit
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
    # Step 2: Trainable entangling layers — this is where quantum advantage lives
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Step 3: Measure expectation value of Pauli-Z on qubit 0 → gives a value in [-1, 1]
    return qml.expval(qml.PauliZ(0))

# Convert the quantum circuit into a PyTorch-trainable layer
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

class HybridQNN(nn.Module):
    """
    Architecture:
      Classical pre-layer (4 → 4) 
      → Quantum VQC layer (4 qubits, 3 entangling layers)
      → Classical post-layer (1 → 1)
      → Sigmoid (output: probability of attack)
    """
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(4, 4)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.post = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.tanh(self.pre(x))       # pre-process features
        x = self.qlayer(x)                 # quantum layer — shape: (batch,)
        x = x.unsqueeze(1)                 # shape: (batch, 1)
        x = self.sigmoid(self.post(x))     # final probability
        return x.squeeze(1)

def get_model():
    return HybridQNN()