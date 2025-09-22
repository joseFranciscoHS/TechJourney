#!/usr/bin/env python3
"""
Demonstrate how sinusoidal encoding modifies volumes
"""

import torch
import numpy as np


def demonstrate_volume_modification():
    """Show exactly how sine/cosine modify each volume"""

    print("How Sinusoidal Encoding Modifies Volumes")
    print("=" * 50)

    # Create a simple example
    batch_size = 2
    num_volumes = 3
    channels = 8  # Small for demonstration
    x, y, z = 2, 2, 2  # Small spatial dimensions

    # Create dummy volume data
    volumes = torch.randn(batch_size, num_volumes, channels, x, y, z)
    volume_indices = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Different volume orders

    print(f"Original volumes shape: {volumes.shape}")
    print(f"Volume indices: {volume_indices}")
    print()

    # Create sinusoidal encoding
    embedding_dim = 8
    num_volumes_encoding = 3

    pe = torch.zeros(num_volumes_encoding, embedding_dim)
    position = torch.arange(0, num_volumes_encoding).unsqueeze(1).float()

    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float()
        * -(torch.log(torch.tensor(10000.0)) / embedding_dim)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    print("Positional Encodings:")
    print("-" * 30)
    for vol in range(num_volumes_encoding):
        print(f"Volume {vol}: {pe[vol, :4].tolist()}")
    print()

    # Show how each volume gets modified
    print("Volume Modification Process:")
    print("-" * 40)

    for batch_idx in range(batch_size):
        print(f"Batch {batch_idx}:")
        for vol_idx in range(num_volumes):
            # Get the volume
            original_vol = volumes[
                batch_idx, vol_idx, :4, 0, 0, 0
            ]  # First 4 channels, first spatial location

            # Get the volume index
            vol_index = volume_indices[batch_idx, vol_idx]

            # Get the positional encoding
            pos_encoding = pe[vol_index, :4]

            # Apply encoding (add to features)
            modified_vol = original_vol + pos_encoding

            print(f"  Volume {vol_idx} (index {vol_index}):")
            print(f"    Original:  {original_vol.tolist()}")
            print(f"    Encoding:  {pos_encoding.tolist()}")
            print(f"    Modified:  {modified_vol.tolist()}")
            print()

    # Show the effect on the model
    print("Why This Helps the Model:")
    print("-" * 30)
    print("1. Each volume gets a unique 'fingerprint'")
    print("2. The model can distinguish between different volume positions")
    print("3. Similar volumes get similar encodings")
    print("4. The encoding provides positional context")
    print()

    # Show volume similarity
    print("Volume Similarity (based on encoding):")
    print("-" * 40)
    for i in range(num_volumes_encoding):
        for j in range(i + 1, num_volumes_encoding):
            similarity = torch.cosine_similarity(pe[i], pe[j], dim=0)
            print(f"Volume {i} vs Volume {j}: {similarity:.4f}")
    print()

    return volumes, pe


if __name__ == "__main__":
    volumes, pe = demonstrate_volume_modification()
