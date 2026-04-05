import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

def safe_hex(val):
    """Convert hex string to int safely."""
    try:
        return int(str(val).strip(), 16)
    except:
        return 0

def byte_entropy(row):
    """Calculate entropy of 8 data bytes."""
    vals = np.array([safe_hex(v) for v in row], dtype=float)
    total = vals.sum()
    if total == 0:
        return 0.0
    vals = vals / total
    return float(-np.sum(vals * np.log2(vals + 1e-9)))

def load_dataset(data_dir="data/"):
    dfs = []
    # FIX: Don't assign label manually — read from Flag column instead
    # Flag column: 'R' = Normal (0), 'T' = Attack (1)
    files = [
        "DoS_dataset.csv",
        "Fuzzy_dataset.csv",
        "RPM_dataset.csv",
        "gear_dataset.csv",
        "normal_dataset.csv",
        "normal_run_data.csv",
    ]
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  Skipping {fname} — not found")
            continue

        df = pd.read_csv(path, header=None, on_bad_lines='skip', engine='python')
        print(f"  {fname}: {len(df)} rows, {len(df.columns)} columns detected")

        # Keep only first 12 columns
        df = df.iloc[:, :12]

        # Pad missing columns with 0
        while len(df.columns) < 12:
            df[len(df.columns)] = 0

        df.columns = ["Timestamp","CAN_ID","DLC",
                      "D0","D1","D2","D3","D4","D5","D6","D7","Flag"]

        # FIX: Label comes from Flag column — 'R' = normal, 'T' = attack
        df["label"] = df["Flag"].apply(
            lambda x: 0 if str(x).strip().upper() == 'R' else 1
        )

        n_normal = (df["label"] == 0).sum()
        n_attack = (df["label"] == 1).sum()
        print(f"    → Normal: {n_normal}, Attack: {n_attack}")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSV files found in data/ folder!")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined)}")
    print(f"Total Normal: {(combined['label']==0).sum()}, "
          f"Total Attack: {(combined['label']==1).sum()}")
    return combined

def extract_features(df):
    print("\nExtracting features...")

    can_id     = df["CAN_ID"].apply(safe_hex).astype(float)
    timestamp  = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(0)
    time_delta = timestamp.diff().fillna(0).abs()
    dlc        = df["DLC"].apply(safe_hex).astype(float)

    data_cols = ["D0","D1","D2","D3","D4","D5","D6","D7"]
    entropy   = df[data_cols].apply(byte_entropy, axis=1)

    features = np.column_stack([can_id, time_delta, dlc, entropy])
    labels   = df["label"].values.astype(int)

    # Remove NaN/inf rows
    mask     = np.isfinite(features).all(axis=1)
    features = features[mask]
    labels   = labels[mask]

    print(f"Features shape: {features.shape}")
    print(f"After cleaning — Normal: {(labels==0).sum()}, Attack: {(labels==1).sum()}")
    return features, labels

def preprocess_and_save(data_dir="data/", out_dir="processed/"):
    os.makedirs(out_dir, exist_ok=True)

    df   = load_dataset(data_dir)
    X, y = extract_features(df)

    if len(X) == 0:
        raise RuntimeError("No valid samples after feature extraction!")

    idx_attack = np.where(y == 1)[0]
    idx_normal = np.where(y == 0)[0]

    if len(idx_attack) == 0 or len(idx_normal) == 0:
        raise RuntimeError(
            f"Need both classes! Got Attack={len(idx_attack)}, Normal={len(idx_normal)}"
        )

    # Balance classes — max 5000 each
    n = min(len(idx_attack), len(idx_normal), 5000)
    idx = np.concatenate([
        np.random.choice(idx_attack, n, replace=False),
        np.random.choice(idx_normal, n, replace=False)
    ])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    print(f"\nBalanced: {n} attack + {n} normal = {len(X)} total samples")

    # Scale to [0, pi]
    scaler = MinMaxScaler(feature_range=(0, 3.14159))
    X      = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )

    np.save(f"{out_dir}/X_train.npy", X_train)
    np.save(f"{out_dir}/X_val.npy",   X_val)
    np.save(f"{out_dir}/X_test.npy",  X_test)
    np.save(f"{out_dir}/y_train.npy", y_train)
    np.save(f"{out_dir}/y_val.npy",   y_val)
    np.save(f"{out_dir}/y_test.npy",  y_test)

    import pickle
    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Done!")
    print(f"   Train : {X_train.shape}")
    print(f"   Val   : {X_val.shape}")
    print(f"   Test  : {X_test.shape}")

if __name__ == "__main__":
    preprocess_and_save()
