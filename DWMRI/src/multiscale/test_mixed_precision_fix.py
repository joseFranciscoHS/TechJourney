#!/usr/bin/env python3
"""
Test script to verify the mixed precision fix for EdgeAwareLoss
"""

import torch
import torch.nn.functional as F
from model import EdgeAwareLoss, MultiScaleDetailNet


def test_mixed_precision_fix():
    """Test that EdgeAwareLoss works with mixed precision training"""
    print("🧪 Testing Mixed Precision Fix for EdgeAwareLoss")
    print("=" * 50)

    # Create model and loss function
    model = MultiScaleDetailNet(
        input_channels=9,
        output_channels=1,
        base_filters=16,  # Smaller for faster testing
        num_volumes=10,
        use_sinusoidal_encoding=True,
    )

    criterion = EdgeAwareLoss(alpha=0.5, beta=0.1)

    # Create sample data
    inputs = torch.randn(2, 9, 16, 16, 16)
    volume_indices = torch.randint(0, 10, (2, 9))
    targets = torch.randn(2, 1, 16, 16, 16)

    print(f"Input shape: {inputs.shape}")
    print(f"Input dtype: {inputs.dtype}")

    # Test 1: Standard precision
    print("\n--- Test 1: Standard Precision ---")
    try:
        outputs = model(inputs, volume_indices)
        loss = criterion(outputs, targets)
        print(f"✅ Standard precision test passed!")
        print(f"Output dtype: {outputs.dtype}")
        print(f"Loss value: {loss.item():.6f}")
    except Exception as e:
        print(f"❌ Standard precision test failed: {e}")
        return False

    # Test 2: Mixed precision
    print("\n--- Test 2: Mixed Precision ---")
    try:
        with torch.cuda.amp.autocast():
            outputs = model(inputs, volume_indices)
            loss = criterion(outputs, targets)

        print(f"✅ Mixed precision test passed!")
        print(f"Output dtype: {outputs.dtype}")
        print(f"Loss value: {loss.item():.6f}")
    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        return False

    # Test 3: Edge detection specifically
    print("\n--- Test 3: Edge Detection with Mixed Precision ---")
    try:
        # Create a sample tensor in half precision
        sample_tensor = torch.randn(1, 1, 16, 16, 16).half()
        print(f"Sample tensor dtype: {sample_tensor.dtype}")

        # Test edge detection
        edges = criterion.detect_edges(sample_tensor)
        print(f"✅ Edge detection with half precision passed!")
        print(f"Edge output dtype: {edges.dtype}")
        print(f"Edge output shape: {edges.shape}")
    except Exception as e:
        print(f"❌ Edge detection test failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 All tests passed! Mixed precision fix is working correctly.")
    return True


if __name__ == "__main__":
    success = test_mixed_precision_fix()
    if not success:
        print("❌ Some tests failed. Please check the implementation.")
        exit(1)
    else:
        print("✅ Mixed precision compatibility verified!")
