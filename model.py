import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
n_qubits = 4
n_layers = 2  

def build_qnn_circuit():
    inputs  = ParameterVector("x", n_qubits)
    weights = ParameterVector("w", n_layers * n_qubits * 3)

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.rx(inputs[i], i)
    w_idx = 0
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qc.rz(weights[w_idx],     qubit)
            qc.ry(weights[w_idx + 1], qubit)
            qc.rz(weights[w_idx + 2], qubit)
            w_idx += 3
        for qubit in range(n_qubits):
            qc.cx(qubit, (qubit + 1) % n_qubits)

    return qc, inputs, weights


def get_model():
    qc, inputs, weights = build_qnn_circuit()

    observable = SparsePauliOp("IIIZ")
    estimator  = Estimator()

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        observables=observable,
        input_params=inputs.params,
        weight_params=weights.params,
    )

    qnn_layer = TorchConnector(qnn)

    class HybridQNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre     = nn.Linear(4, 4)
            self.qlayer  = qnn_layer
            self.post    = nn.Linear(1, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.tanh(self.pre(x))
            x = self.qlayer(x)
            x = self.sigmoid(self.post(x))
            return x.squeeze(1)

    return HybridQNN()


def get_circuit_diagram():
    qc, _, _ = build_qnn_circuit()
    return qc
