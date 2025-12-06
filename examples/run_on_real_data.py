
import os
import torch
import numpy as np
from datasets import load_dataset
from pulsar.model import PULSAR
from pulsar.utils import extract_donor_embeddings_from_h5ad
import scanpy as sc
import anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def run_on_real_data():
    print("==================================================")
    print("PART 1: DOWNSTREAM TASK ON REAL DATA (Zero-Shot)")
    print("==================================================")
    print("Loading real data from HuggingFace (KuanP/PULSAR_DONORxEMBED_zero_shot)...")

    # 1. Load DONORxEMBED zero-shot dataset
    try:
        ds = load_dataset("KuanP/PULSAR_DONORxEMBED_zero_shot", split="train")
        print(f"Dataset loaded. Number of samples: {len(ds)}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 2. Inspect labels
    labels = ds["disease_label"]
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nDisease label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")

    # 3. Prepare data for classification (Normal vs COVID-19)
    # We filter for these two classes to make a clean binary classification demo.
    target_classes = ["normal", "COVID-19"]
    print(f"\nPreparing data for classification: {target_classes[0]} vs {target_classes[1]}...")

    X = []
    y = []

    for item in ds:
        label = item["disease_label"]
        if label in target_classes:
            X.append(item["embedding"])
            y.append(1 if label == "COVID-19" else 0)

    X = np.array(X)
    y = np.array(y)

    print(f"Filtered dataset size: {len(X)}")
    print(f"Feature shape: {X.shape}")

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 5. Train Classifier
    print("\nTraining Logistic Regression classifier on PULSAR embeddings...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "COVID-19"]))

    print("\n==================================================")
    print("PART 2: DEMONSTRATE EMBEDDING EXTRACTION FLOW")
    print("==================================================")
    print("(Simulating raw cell-level input to PULSAR model)")

    print("\nLoading PULSAR-pbmc model...")
    try:
        model = PULSAR.from_pretrained("KuanP/PULSAR-pbmc")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\nCreating mock cell-level data (AnnData)...")

    # Create mock AnnData to simulate raw scRNA-seq input
    n_obs = 500  # Number of cells
    n_vars = 1280 # PULSAR input size (UCE embedding size)

    # Generate random cell embeddings
    X_uce = np.random.randn(n_obs, n_vars).astype(np.float32)

    # Assign cells to 5 donors
    donor_ids = [f"donor_{i%5}" for i in range(n_obs)]

    obs = {
        "donor_id": donor_ids,
        "sample_id": [f"sample_{i}" for i in range(n_obs)]
    }

    adata = ad.AnnData(X=X_uce, obs=obs)
    adata.obsm["X_uce"] = X_uce # Key used by extract_donor_embeddings_from_h5ad

    print(f"Mock AnnData created with {n_obs} cells from 5 donors.")

    # Run extraction
    print("Running extract_donor_embeddings_from_h5ad...")
    # We use CPU for the example to ensure it runs everywhere
    donor_embeddings = extract_donor_embeddings_from_h5ad(
        adata=adata,
        model=model,
        donor_id_key="donor_id",
        embedding_key="X_uce",
        sample_cell_num=100, # Cells per donor
        device="cpu",
        use_tqdm=False
    )

    print(f"\nSuccessfully extracted embeddings for {len(donor_embeddings)} donors.")
    for donor_id, data in donor_embeddings.items():
        emb_shape = np.array(data['embedding'][0]).shape
        print(f"  {donor_id}: Embedding shape {emb_shape}")

if __name__ == "__main__":
    run_on_real_data()
