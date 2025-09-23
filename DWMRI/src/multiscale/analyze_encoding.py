#!/usr/bin/env python3
"""
Analyze sinusoidal positional encoding dimensions
"""

import torch
import numpy as np


def analyze_positional_encoding():
    """Analyze what each dimension of positional encoding represents"""

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

    # Analyze frequency distribution
    print("Frequency Distribution:")
    print("-" * 30)
    frequencies = []
    for j in range(embedding_dim // 2):
        freq = 1.0 / (10000 ** (2 * j / embedding_dim))
        frequencies.append(freq)

    print(f"Lowest frequency:  {min(frequencies):.6f}")
    print(f"Highest frequency: {max(frequencies):.6f}")
    print(f"Frequency range:   {max(frequencies) - min(frequencies):.6f}")
    print()

    # Show how dimensions capture different patterns
    print("Pattern Analysis:")
    print("-" * 30)
    print("Low frequency dimensions (0-7): Capture broad volume relationships")
    print("Medium frequency dimensions (8-31): Capture moderate volume patterns")
    print("High frequency dimensions (32-63): Capture fine-grained volume details")
    print()

    return pe


if __name__ == "__main__":
    pe = analyze_positional_encoding()
