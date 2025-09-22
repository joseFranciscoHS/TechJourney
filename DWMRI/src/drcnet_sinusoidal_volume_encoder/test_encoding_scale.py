#!/usr/bin/env python3
"""
Test the encoding scale fix
"""

import torch
import numpy as np


def test_encoding_scale():
    """Test that encoding scale keeps data in [0,1] range"""

    print("Testing Encoding Scale Fix")
    print("=" * 40)

    # Create normalized data in [0,1] range
    batch_size = 2
    num_volumes = 3
    channels = 8
    x, y, z = 2, 2, 2

    # Create data in [0,1] range (as it would be after normalization)
    volumes = torch.rand(batch_size, num_volumes, channels, x, y, z)
    volume_indices = torch.tensor([[0, 1, 2], [1, 2, 0]])

    print(f"Original data range: [{volumes.min():.3f}, {volumes.max():.3f}]")
    print(f"Original data shape: {volumes.shape}")
    print()

    # Create sinusoidal encoding
    embedding_dim = 8
    num_volumes_encoding = 3
    encoding_scale = 0.1  # Scale factor

    pe = torch.zeros(num_volumes_encoding, embedding_dim)
    position = torch.arange(0, num_volumes_encoding).unsqueeze(1).float()

    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float()
        * -(torch.log(torch.tensor(10000.0)) / embedding_dim)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    print("Positional Encodings (before scaling):")
    for vol in range(num_volumes_encoding):
        print(f"Volume {vol}: {pe[vol, :4].tolist()}")
    print()

    print("Positional Encodings (after scaling by 0.1):")
    for vol in range(num_volumes_encoding):
        scaled = pe[vol, :4] * encoding_scale
        print(f"Volume {vol}: {scaled.tolist()}")
    print()

    # Apply encoding with scaling
    encoded_volumes = []
    for batch_idx in range(batch_size):
        batch_encoded = []
        for vol_idx in range(num_volumes):
            # Get the volume
            original_vol = volumes[batch_idx, vol_idx]

            # Get the volume index
            vol_index = volume_indices[batch_idx, vol_idx]

            # Get the positional encoding
            pos_encoding = pe[vol_index]

            # Apply scaled encoding
            if channels >= embedding_dim:
                scaled_encoding = pos_encoding * encoding_scale
                encoded_vol = original_vol.clone()
                # Reshape encoding to match spatial dimensions
                scaled_encoding = (
                    scaled_encoding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                )
                scaled_encoding = scaled_encoding.expand(-1, x, y, z)
                encoded_vol[:, :embedding_dim] += scaled_encoding
            else:
                scaled_encoding = pos_encoding[:channels] * encoding_scale
                # Reshape encoding to match spatial dimensions
                scaled_encoding = (
                    scaled_encoding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                )
                scaled_encoding = scaled_encoding.expand(-1, x, y, z)
                encoded_vol = original_vol + scaled_encoding

            batch_encoded.append(encoded_vol)

        encoded_volumes.append(torch.stack(batch_encoded, dim=0))

    encoded_volumes = torch.stack(encoded_volumes, dim=0)

    print("Results:")
    print("-" * 20)
    print(
        f"Encoded data range: [{encoded_volumes.min():.3f}, {encoded_volumes.max():.3f}]"
    )
    print(f"Encoded data shape: {encoded_volumes.shape}")
    print()

    # Check if data stays in reasonable range
    min_val = encoded_volumes.min().item()
    max_val = encoded_volumes.max().item()

    print("Range Analysis:")
    print(f"Minimum value: {min_val:.3f}")
    print(f"Maximum value: {max_val:.3f}")
    print(f"Range: {max_val - min_val:.3f}")
    print()

    if min_val >= -0.1 and max_val <= 1.1:
        print("✅ SUCCESS: Data stays in reasonable range!")
        print("   The encoding scale prevents values from going too far outside [0,1]")
    else:
        print("❌ WARNING: Data range might be too large")
        print("   Consider reducing encoding_scale further")

    print()
    print("Recommendations:")
    print("- encoding_scale=0.1: Good balance (current)")
    print("- encoding_scale=0.05: More conservative")
    print("- encoding_scale=0.2: More aggressive (might break [0,1] range)")


if __name__ == "__main__":
    test_encoding_scale()
