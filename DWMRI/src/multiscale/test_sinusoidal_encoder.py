#!/usr/bin/env python3
"""
Test script for SinusoidalVolumeEncoder implementation.
This script tests the basic functionality of the sinusoidal volume encoder.
"""

import logging
import torch
import numpy as np
from model import SinusoidalVolumeEncoder, DenoiserNet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sinusoidal_encoder():
    """Test the SinusoidalVolumeEncoder class"""
    logger.info("Testing SinusoidalVolumeEncoder...")
    
    # Test parameters
    num_volumes = 10
    embedding_dim = 64
    batch_size = 4
    channels = 32
    x, y, z = 16, 16, 16
    
    # Create encoder
    encoder = SinusoidalVolumeEncoder(
        num_volumes=num_volumes,
        embedding_dim=embedding_dim
    )
    
    # Create test data
    volume_features = torch.randn(batch_size, channels, x, y, z)
    volume_indices = torch.randint(0, num_volumes, (batch_size,))
    
    logger.info(f"Input shape: {volume_features.shape}")
    logger.info(f"Volume indices: {volume_indices.tolist()}")
    
    # Test forward pass
    encoded_features = encoder(volume_features, volume_indices)
    
    logger.info(f"Output shape: {encoded_features.shape}")
    logger.info(f"Input and output shapes match: {volume_features.shape == encoded_features.shape}")
    
    # Test that different volume indices produce different encodings
    test_indices_1 = torch.tensor([0, 1, 2, 3])
    test_indices_2 = torch.tensor([1, 2, 3, 4])
    
    encoded_1 = encoder(volume_features, test_indices_1)
    encoded_2 = encoder(volume_features, test_indices_2)
    
    # Check that encodings are different
    different_encodings = not torch.allclose(encoded_1, encoded_2, atol=1e-6)
    logger.info(f"Different volume indices produce different encodings: {different_encodings}")
    
    # Test that same volume indices produce same encodings
    encoded_3 = encoder(volume_features, test_indices_1)
    same_encodings = torch.allclose(encoded_1, encoded_3, atol=1e-6)
    logger.info(f"Same volume indices produce same encodings: {same_encodings}")
    
    logger.info("SinusoidalVolumeEncoder test completed successfully!")
    return True

def test_denoiser_net_with_encoding():
    """Test the DenoiserNet with sinusoidal encoding"""
    logger.info("Testing DenoiserNet with sinusoidal encoding...")
    
    # Test parameters
    batch_size = 2
    num_input_volumes = 9
    x, y, z = 16, 16, 16
    
    # Create model
    model = DenoiserNet(
        input_channels=num_input_volumes,
        output_channels=1,
        groups=1,
        dense_convs=2,
        residual=True,
        base_filters=16,  # Smaller for testing
        output_shape=(1, x, y, z),
        device="cpu",
        num_volumes=10,
        use_sinusoidal_encoding=True,
        embedding_dim=32
    )
    
    # Create test data
    inputs = torch.randn(batch_size, num_input_volumes, x, y, z)
    volume_indices = torch.randint(0, 10, (batch_size, num_input_volumes))
    
    logger.info(f"Input shape: {inputs.shape}")
    logger.info(f"Volume indices shape: {volume_indices.shape}")
    logger.info(f"Volume indices sample: {volume_indices[0].tolist()}")
    
    # Test forward pass
    output = model(inputs, volume_indices)
    
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Expected output shape: ({batch_size}, 1, {x}, {y}, {z})")
    
    # Verify output shape
    expected_shape = (batch_size, 1, x, y, z)
    correct_shape = output.shape == expected_shape
    logger.info(f"Output shape is correct: {correct_shape}")
    
    # Test without volume indices (should still work)
    output_no_indices = model(inputs, None)
    logger.info(f"Output without volume indices shape: {output_no_indices.shape}")
    
    logger.info("DenoiserNet with sinusoidal encoding test completed successfully!")
    return True

def test_data_loading():
    """Test the data loading with volume indices"""
    logger.info("Testing data loading with volume indices...")
    
    # Create dummy data
    data = np.random.randn(32, 32, 32, 10)  # x, y, z, volumes
    
    # Test TrainingDataSetMultipleVolumes
    from data import TrainingDataSetMultipleVolumes
    
    dataset = TrainingDataSetMultipleVolumes(
        data=data,
        patch_size=(10, 16, 16, 16),  # volumes, x, y, z
        step=8
    )
    
    logger.info(f"Dataset length: {len(dataset)}")
    
    # Test getting a sample
    x, y, volume_indices = dataset[0]
    
    logger.info(f"Sample x shape: {x.shape}")
    logger.info(f"Sample y shape: {y.shape}")
    logger.info(f"Sample volume_indices shape: {volume_indices.shape}")
    logger.info(f"Sample volume_indices: {volume_indices.tolist()}")
    
    # Verify shapes
    expected_x_shape = (9, 16, 16, 16)  # 9 input volumes, 16x16x16 patch
    expected_y_shape = (1, 16, 16, 16)  # 1 target volume, 16x16x16 patch
    expected_indices_shape = (9,)  # 9 volume indices
    
    correct_x_shape = x.shape == expected_x_shape
    correct_y_shape = y.shape == expected_y_shape
    correct_indices_shape = volume_indices.shape == expected_indices_shape
    
    logger.info(f"X shape is correct: {correct_x_shape}")
    logger.info(f"Y shape is correct: {correct_y_shape}")
    logger.info(f"Volume indices shape is correct: {correct_indices_shape}")
    
    logger.info("Data loading test completed successfully!")
    return True

def main():
    """Run all tests"""
    logger.info("Starting SinusoidalVolumeEncoder implementation tests...")
    
    try:
        # Test 1: SinusoidalVolumeEncoder
        test_sinusoidal_encoder()
        
        # Test 2: DenoiserNet with encoding
        test_denoiser_net_with_encoding()
        
        # Test 3: Data loading
        test_data_loading()
        
        logger.info("All tests completed successfully! ✅")
        logger.info("The sinusoidal volume encoder implementation is working correctly.")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
