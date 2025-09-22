#!/usr/bin/env python3
"""
Visualize sinusoidal positional encoding dimensions
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_positional_encoding():
    """Visualize what each dimension of positional encoding represents"""

    num_volumes = 10
    embedding_dim = 64
    max_freq = 10000.0

    # Create sinusoidal encodings (same as in our model)
    pe = torch.zeros(num_volumes, embedding_dim)
    position = torch.arange(0, num_volumes).unsqueeze(1).float()

    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float()
        * -(torch.log(torch.tensor(max_freq)) / embedding_dim)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    print("Positional Encoding Analysis")
    print("=" * 50)
    print(f"Number of volumes: {num_volumes}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Total encoding size: {pe.shape}")
    print()

    # Analyze different frequency components
    print("Frequency Components:")
    print("-" * 30)
    for j in range(min(8, embedding_dim // 2)):
        freq = 1.0 / (10000 ** (2 * j / embedding_dim))
        print(f"Dimension {2*j:2d}: sin(pos * {freq:.6f}) - Frequency: {freq:.6f}")
        print(f"Dimension {2*j+1:2d}: cos(pos * {freq:.6f}) - Frequency: {freq:.6f}")
        print()

    # Show actual values for first few volumes
    print("Actual Encoding Values (first 8 dimensions):")
    print("-" * 50)
    for vol in range(min(5, num_volumes)):
        print(f"Volume {vol}: {pe[vol, :8].tolist()}")
    print()

    # Show how different volumes have different patterns
    print("Volume Similarity Analysis:")
    print("-" * 30)
    for i in range(min(3, num_volumes)):
        for j in range(i + 1, min(4, num_volumes)):
            similarity = torch.cosine_similarity(pe[i], pe[j], dim=0)
            print(f"Volume {i} vs Volume {j}: {similarity:.4f}")
    print()

    return pe


if __name__ == "__main__":
    pe = visualize_positional_encoding()

    # Create a simple visualization
    plt.figure(figsize=(12, 8))

    # Plot first 8 dimensions for all volumes
    plt.subplot(2, 2, 1)
    for dim in range(0, 8, 2):
        plt.plot(range(10), pe[:, dim].numpy(), label=f"Dim {dim} (sin)")
    plt.title("Sine Components (Low Frequencies)")
    plt.xlabel("Volume Position")
    plt.ylabel("Encoding Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for dim in range(1, 8, 2):
        plt.plot(range(10), pe[:, dim].numpy(), label=f"Dim {dim} (cos)")
    plt.title("Cosine Components (Low Frequencies)")
    plt.xlabel("Volume Position")
    plt.ylabel("Encoding Value")
    plt.legend()
    plt.grid(True)

    # Plot higher frequency components
    plt.subplot(2, 2, 3)
    for dim in range(16, 24, 2):
        plt.plot(range(10), pe[:, dim].numpy(), label=f"Dim {dim} (sin)")
    plt.title("Sine Components (Higher Frequencies)")
    plt.xlabel("Volume Position")
    plt.ylabel("Encoding Value")
    plt.legend()
    plt.grid(True)

    # Heatmap of all encodings
    plt.subplot(2, 2, 4)
    plt.imshow(pe.numpy(), aspect="auto", cmap="viridis")
    plt.title("Positional Encoding Heatmap")
    plt.xlabel("Dimension")
    plt.ylabel("Volume Position")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("positional_encoding_analysis.png", dpi=150, bbox_inches="tight")
    print("Visualization saved as 'positional_encoding_analysis.png'")
    plt.show()

if __name__ == "__main__":
    pe = visualize_positional_encoding()
