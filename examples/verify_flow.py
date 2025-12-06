
import torch
import numpy as np
from pulsar.model import PULSARConfig, PULSAR, PULSARForClassification
from pulsar.utils import collate_fn

def verify_pulsar_flow():
    print("Verifying PULSAR flow...")

    # 1. Setup Config
    # Ensure hidden_size is divisible by num_attention_heads
    config = PULSARConfig(
        input_size=128,  # Mock embedding size
        hidden_size=64,
        seq_length=10,   # Small seq length for test
        encoder_num_hidden_layers=2,
        encoder_num_attention_heads=4,
        encoder_intermediate_size=256,
        decoder_num_hidden_layers=2,  # Also set for decoder
        decoder_num_attention_heads=4, # Ensure compatibility
        decoder_intermediate_size=256,
        num_labels=2
    )

    # 2. Instantiate Model
    model = PULSARForClassification(config)
    print("Model instantiated.")

    # 3. Mock Data
    # Batch of 2 donors
    # Donor 1 has 15 cells, Donor 2 has 5 cells
    embedding_dim = 128

    batch = [
        {
            "donor_id": "donor1",
            "cell_embedding": np.random.randn(15, embedding_dim).astype(np.float32),
            "cell_type_idx": [-1] * 15,
            "labels": 1
        },
        {
            "donor_id": "donor2",
            "cell_embedding": np.random.randn(5, embedding_dim).astype(np.float32),
            "cell_type_idx": [-1] * 5,
            "labels": 0
        }
    ]

    # 4. Collate (Sampling)
    max_length = 10
    collated = collate_fn(batch, max_length=max_length, resample=True)

    cell_embeddings = collated["cell_embedding"]
    cell_type_idx = collated["cell_type_idx"]
    labels = collated["labels"]

    print(f"Collated cell_embeddings shape: {cell_embeddings.shape}")
    # Expected: (2, 10, 128)

    # 5. Forward Pass
    output = model(
        cell_embedding=cell_embeddings,
        cell_type_idx=cell_type_idx,
        labels=labels
    )

    print(f"Loss: {output.loss}")
    print(f"Logits shape: {output.logits.shape}")
    # Expected: (2, 2)
    print(f"CLS embedding shape: {output.cls_embedding.shape}")
    # Expected: (2, 64)

if __name__ == "__main__":
    verify_pulsar_flow()
