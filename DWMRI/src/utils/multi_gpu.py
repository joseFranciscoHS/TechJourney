"""
Multi-GPU training utilities for medical image processing models.

This module provides flexible multi-GPU setup functionality that can be used
across different models (DRCNet, MDS2S, P2S, etc.) with automatic learning
rate scaling and fallback support.
"""

import logging
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel


class MultiGPUConfig:
    """Configuration class for multi-GPU training settings."""
    
    def __init__(
        self,
        multi_gpu: bool = True,
        gpu_ids: List[int] = None,
        auto_scale_lr: bool = True,
        base_learning_rate: float = 0.0005,
        batch_size_per_gpu: int = 16,
    ):
        """
        Initialize multi-GPU configuration.
        
        Args:
            multi_gpu: Enable multi-GPU training
            gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2])
            auto_scale_lr: Automatically scale learning rate based on GPU count
            base_learning_rate: Base learning rate for single GPU
            batch_size_per_gpu: Batch size per GPU
        """
        self.multi_gpu = multi_gpu
        self.gpu_ids = gpu_ids or [0, 1, 2]
        self.auto_scale_lr = auto_scale_lr
        self.base_learning_rate = base_learning_rate
        self.batch_size_per_gpu = batch_size_per_gpu


def setup_multi_gpu(
    model: nn.Module,
    config: MultiGPUConfig,
    verbose: bool = True
) -> Tuple[nn.Module, float, int]:
    """
    Setup multi-GPU training with automatic learning rate scaling.
    
    Args:
        model: PyTorch model to wrap
        config: MultiGPUConfig object with settings
        verbose: Enable detailed logging
        
    Returns:
        Tuple of (wrapped_model, effective_learning_rate, effective_batch_size)
    """
    if verbose:
        logging.info("Setting up multi-GPU training...")
    
    if not config.multi_gpu:
        if verbose:
            logging.info("Single GPU training mode")
        return model, config.base_learning_rate, config.batch_size_per_gpu
    
    available_gpus = torch.cuda.device_count()
    requested_gpus = len(config.gpu_ids)
    
    # Validate GPU availability
    if available_gpus < requested_gpus:
        if verbose:
            logging.warning(f"Only {available_gpus} GPUs available, requested {requested_gpus}")
        actual_gpus = min(available_gpus, requested_gpus)
        gpu_ids = list(range(actual_gpus))
    else:
        gpu_ids = config.gpu_ids
    
    # Wrap model with DataParallel
    if len(gpu_ids) > 1:
        if verbose:
            logging.info(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
            logging.info(f"Available GPUs: {available_gpus}")
        
        model = DataParallel(model, device_ids=gpu_ids)
        
        # Calculate effective batch size
        effective_batch_size = config.batch_size_per_gpu * len(gpu_ids)
        
        # Scale learning rate based on number of GPUs
        if config.auto_scale_lr:
            scale_factor = len(gpu_ids) ** 0.5  # Square root scaling
            effective_lr = config.base_learning_rate * scale_factor
            if verbose:
                logging.info(f"Learning rate scaled: {config.base_learning_rate} -> {effective_lr:.6f} (factor: {scale_factor:.2f})")
                logging.info(f"Effective batch size: {config.batch_size_per_gpu} -> {effective_batch_size} (per GPU × {len(gpu_ids)} GPUs)")
            return model, effective_lr, effective_batch_size
        else:
            if verbose:
                logging.info(f"Learning rate scaling disabled, using base LR: {config.base_learning_rate}")
                logging.info(f"Effective batch size: {effective_batch_size}")
            return model, config.base_learning_rate, effective_batch_size
    else:
        if verbose:
            logging.info("Single GPU training (multi_gpu enabled but only 1 GPU available)")
        return model, config.base_learning_rate, config.batch_size_per_gpu


def get_gpu_info() -> dict:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    gpu_count = torch.cuda.device_count()
    gpu_info = {
        "count": gpu_count,
        "devices": []
    }
    
    for i in range(gpu_count):
        device_props = torch.cuda.get_device_properties(i)
        gpu_info["devices"].append({
            "id": i,
            "name": device_props.name,
            "memory_total": device_props.total_memory,
            "memory_total_gb": device_props.total_memory / (1024**3),
            "compute_capability": f"{device_props.major}.{device_props.minor}"
        })
    
    return gpu_info


def log_gpu_info():
    """Log detailed GPU information."""
    gpu_info = get_gpu_info()
    logging.info(f"GPU Information:")
    logging.info(f"  Total GPUs: {gpu_info['count']}")
    
    for device in gpu_info["devices"]:
        logging.info(f"  GPU {device['id']}: {device['name']}")
        logging.info(f"    Memory: {device['memory_total_gb']:.1f} GB")
        logging.info(f"    Compute Capability: {device['compute_capability']}")


def create_multi_gpu_config_from_dict(config_dict: dict) -> MultiGPUConfig:
    """
    Create MultiGPUConfig from dictionary (e.g., from YAML config).
    
    Args:
        config_dict: Dictionary with multi-GPU settings
        
    Returns:
        MultiGPUConfig object
    """
    return MultiGPUConfig(
        multi_gpu=config_dict.get("multi_gpu", True),
        gpu_ids=config_dict.get("gpu_ids", [0, 1, 2]),
        auto_scale_lr=config_dict.get("auto_scale_lr", True),
        base_learning_rate=config_dict.get("learning_rate", 0.0005),
        batch_size_per_gpu=config_dict.get("batch_size", 16),
    )


def validate_multi_gpu_setup(config: MultiGPUConfig) -> bool:
    """
    Validate multi-GPU configuration.
    
    Args:
        config: MultiGPUConfig to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    if not config.multi_gpu:
        return True
    
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        logging.error("No CUDA GPUs available")
        return False
    
    if not config.gpu_ids:
        logging.error("GPU IDs list is empty")
        return False
    
    invalid_gpu_ids = [gpu_id for gpu_id in config.gpu_ids if gpu_id >= available_gpus]
    if invalid_gpu_ids:
        logging.error(f"Invalid GPU IDs: {invalid_gpu_ids}. Available GPUs: 0-{available_gpus-1}")
        return False
    
    return True


# Convenience functions for common configurations
def create_single_gpu_config(base_lr: float = 0.0005, batch_size: int = 16) -> MultiGPUConfig:
    """Create configuration for single GPU training."""
    return MultiGPUConfig(
        multi_gpu=False,
        base_learning_rate=base_lr,
        batch_size_per_gpu=batch_size,
    )


def create_multi_gpu_config(
    gpu_ids: List[int] = None,
    base_lr: float = 0.0005,
    batch_size: int = 16,
    auto_scale_lr: bool = True
) -> MultiGPUConfig:
    """Create configuration for multi-GPU training."""
    return MultiGPUConfig(
        multi_gpu=True,
        gpu_ids=gpu_ids or [0, 1, 2],
        base_learning_rate=base_lr,
        batch_size_per_gpu=batch_size,
        auto_scale_lr=auto_scale_lr,
    )
