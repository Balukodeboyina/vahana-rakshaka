import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle, json
from model import get_model

def load_data(out_dir="processed/"):
    X_train = np.load(f"{out_dir}/X_train.npy").astype(np.float32)
    X_val   = np.load(f"{out_dir}/X_val.npy").astype(np.float32)
    X_test  = np.load(f"{out_dir}/X_test.npy").astype(np.float32)
    y_train = np.load(f"{out_dir}/y_train.npy").astype(np.float32)
    y_val   = np.load(f"{out_dir}/y_val.npy").astype(np.float32)
    y_test  = np.load(f"{out_dir}/y_test.npy").astype(np.float32)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_quantum_model(X_train, y_train, X_val, y_val, epochs=30):
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = (model(torch.tensor(X_val)) > 0.5).float().numpy()
        acc = accuracy_score(y_val, val_pred)
        f1  = f1_score(y_val, val_pred)
        history.append({"epoch": epoch+1, "loss": total_loss/len(loader),
                         "val_acc": acc, "val_f1": f1})
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.4f} "
              f"| Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
    torch.save(model.state_dict(), "processed/qml_model.pt")
    return model, history

def train_classical_baselines(X_train, y_train, X_test, y_test):
    results = {}
    # SVM
    svm = SVC(kernel="rbf", C=1.0, probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results["SVM"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "false_neg_rate": float(np.mean((y_test==1) & (y_pred==0)))
    }
    with open("processed/svm.pkl","wb") as f: pickle.dump(svm, f)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "false_neg_rate": float(np.mean((y_test==1) & (y_pred==0)))
    }
    with open("processed/rf.pkl","wb") as f: pickle.dump(rf, f)
    return results

if __name__ == "__main__":
    X_train,X_val,X_test,y_train,y_val,y_test = load_data()
    print("Training Hybrid QML model...")
    model, history = train_quantum_model(X_train,y_train,X_val,y_val, epochs=30)
    # QML test metrics
    model.eval()
    with torch.no_grad():
        qml_pred = (model(torch.tensor(X_test)) > 0.5).float().numpy()
    qml_results = {
        "QML-IDS (Ours)": {
            "accuracy": accuracy_score(y_test, qml_pred),
            "f1": f1_score(y_test, qml_pred),
            "false_neg_rate": float(np.mean((y_test==1) & (qml_pred==0)))
        }
    }
    print("\nTraining classical baselines...")
    classical = train_classical_baselines(X_train,y_train,X_test,y_test)
    all_results = {**qml_results, **classical}
    with open("processed/benchmark.json","w") as f:
        json.dump(all_results, f, indent=2)
    print("\n=== BENCHMARK RESULTS ===")
    for name, r in all_results.items():
        print(f"{name}: Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | FNR={r['false_neg_rate']:.4f}")