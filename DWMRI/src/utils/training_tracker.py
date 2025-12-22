"""
Training Loss Tracker for DWMRI models.

This module provides utilities for tracking and visualizing training losses
across epochs for different models.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class TrainingLossTracker:
    """
    Tracks training losses per epoch and saves them to JSON files.
    Supports checkpoint resuming and provides plotting functionality.
    """

    def __init__(self, loss_dir: str):
        """
        Initialize the training loss tracker.

        Args:
            loss_dir: Directory path where losses.json will be saved
        """
        self.loss_dir = loss_dir
        self.losses_file = os.path.join(loss_dir, "losses.json")

        # Initialize loss history
        self.loss_history = {
            "epochs": [],
            "best_loss": float("inf"),
            "best_epoch": -1,
        }

        # Create directory if it doesn't exist
        os.makedirs(loss_dir, exist_ok=True)

        # Load existing losses if file exists (for resuming)
        if os.path.exists(self.losses_file):
            self.load_existing()
            logging.info(
                f"Loaded existing losses from {self.losses_file} - "
                f"{len(self.loss_history['epochs'])} epochs, "
                f"best loss: {self.loss_history['best_loss']:.6f} at epoch {self.loss_history['best_epoch']}"
            )

    def record_epoch(
        self,
        epoch: int,
        avg_loss: float,
        min_loss: float,
        max_loss: float,
        std_loss: float,
        learning_rate: float,
    ):
        """
        Record loss metrics for a single epoch.

        Args:
            epoch: Epoch number (1-indexed)
            avg_loss: Average loss for the epoch
            min_loss: Minimum loss in the epoch
            max_loss: Maximum loss in the epoch
            std_loss: Standard deviation of losses in the epoch
            learning_rate: Learning rate used in this epoch
        """
        epoch_data = {
            "epoch": epoch,
            "avg_loss": float(avg_loss),
            "min_loss": float(min_loss),
            "max_loss": float(max_loss),
            "std_loss": float(std_loss),
            "learning_rate": float(learning_rate),
        }

        # Check if this epoch already exists (for resuming)
        existing_idx = None
        for idx, existing_epoch in enumerate(self.loss_history["epochs"]):
            if existing_epoch["epoch"] == epoch:
                existing_idx = idx
                break

        if existing_idx is not None:
            # Update existing epoch
            self.loss_history["epochs"][existing_idx] = epoch_data
            logging.debug(f"Updated epoch {epoch} in loss history")
        else:
            # Add new epoch
            self.loss_history["epochs"].append(epoch_data)
            logging.debug(f"Added epoch {epoch} to loss history")

        # Update best loss
        if avg_loss < self.loss_history["best_loss"]:
            self.loss_history["best_loss"] = float(avg_loss)
            self.loss_history["best_epoch"] = epoch

        # Save after each epoch
        self.save()

    def load_existing(self):
        """Load existing loss history from JSON file."""
        try:
            with open(self.losses_file, "r") as f:
                self.loss_history = json.load(f)
            logging.info(
                f"Loaded {len(self.loss_history['epochs'])} epochs from existing losses file"
            )
        except Exception as e:
            logging.warning(f"Failed to load existing losses: {e}. Starting fresh.")
            self.loss_history = {
                "epochs": [],
                "best_loss": float("inf"),
                "best_epoch": -1,
            }

    def save(self):
        """Save current loss history to JSON file."""
        try:
            with open(self.losses_file, "w") as f:
                json.dump(self.loss_history, f, indent=2)
            logging.debug(f"Saved losses to {self.losses_file}")
        except Exception as e:
            logging.error(f"Failed to save losses: {e}")

    def get_best_loss(self) -> Tuple[float, int]:
        """
        Get the best loss and corresponding epoch.

        Returns:
            Tuple of (best_loss, best_epoch)
        """
        return self.loss_history["best_loss"], self.loss_history["best_epoch"]

    def get_loss_history(self) -> Dict[str, Any]:
        """
        Get the complete loss history.

        Returns:
            Dictionary containing epochs, best_loss, and best_epoch
        """
        return self.loss_history.copy()

    @staticmethod
    def load_losses_file(losses_file_path: str) -> Dict[str, Any]:
        """
        Load a losses.json file.

        Args:
            losses_file_path: Path to the losses.json file

        Returns:
            Dictionary containing the loss history
        """
        with open(losses_file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def plot_losses(
        losses_file_path: str,
        save_path: Optional[str] = None,
        show: bool = True,
        plot_lr: bool = True,
        plot_range: bool = True,
    ):
        """
        Plot training losses from a single losses.json file.

        Args:
            losses_file_path: Path to the losses.json file
            save_path: Optional path to save the plot (if None, not saved)
            show: Whether to display the plot
            plot_lr: Whether to plot learning rate on secondary y-axis
            plot_range: Whether to plot min/max range as shaded region
        """
        # Load losses
        loss_data = TrainingLossTracker.load_losses_file(losses_file_path)
        epochs_data = loss_data["epochs"]

        if not epochs_data:
            logging.warning(f"No epoch data found in {losses_file_path}")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(epochs_data)

        # Create figure with subplots
        if plot_lr:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None

        # Plot average loss
        ax1.plot(df["epoch"], df["avg_loss"], "b-", linewidth=2, label="Average Loss")

        # Plot min/max range if requested
        if plot_range:
            ax1.fill_between(
                df["epoch"],
                df["min_loss"],
                df["max_loss"],
                alpha=0.2,
                color="blue",
                label="Min/Max Range",
            )

        # Mark best epoch
        best_epoch = loss_data.get("best_epoch", -1)
        best_loss = loss_data.get("best_loss", float("inf"))
        if best_epoch > 0:
            ax1.axvline(
                x=best_epoch,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Best Epoch ({best_epoch})",
            )
            ax1.plot(
                best_epoch,
                best_loss,
                "ro",
                markersize=10,
                label=f"Best Loss: {best_loss:.6f}",
            )

        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        # Plot learning rate if requested
        if plot_lr and ax2 is not None:
            ax2.plot(
                df["epoch"],
                df["learning_rate"],
                "g-",
                linewidth=2,
                label="Learning Rate",
            )
            ax2.set_xlabel("Epoch", fontsize=12)
            ax2.set_ylabel("Learning Rate", fontsize=12)
            ax2.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
            ax2.set_yscale("log")  # Log scale for learning rate

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved loss plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_multiple_losses(
        losses_file_paths: List[str],
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        plot_lr: bool = False,
        plot_range: bool = False,
    ):
        """
        Plot training losses from multiple losses.json files for comparison.

        Args:
            losses_file_paths: List of paths to losses.json files
            labels: Optional list of labels for each file (if None, auto-generated from paths)
            save_path: Optional path to save the plot (if None, not saved)
            show: Whether to display the plot
            plot_lr: Whether to plot learning rate curves (can be cluttered with multiple runs)
            plot_range: Whether to plot min/max range (can be cluttered with multiple runs)
        """
        if not losses_file_paths:
            logging.warning("No loss files provided for plotting")
            return

        # Generate labels if not provided
        if labels is None:
            labels = []
            for path in losses_file_paths:
                # Extract meaningful part of path (e.g., learning_rate_0.0001)
                path_parts = path.split(os.sep)
                # Try to find the most descriptive part
                label = (
                    path_parts[-2]
                    if len(path_parts) > 1
                    else os.path.basename(os.path.dirname(path))
                )
                labels.append(label)

        if len(labels) != len(losses_file_paths):
            raise ValueError("Number of labels must match number of loss files")

        # Load all losses
        all_data = []
        for path in losses_file_paths:
            try:
                data = TrainingLossTracker.load_losses_file(path)
                df = pd.DataFrame(data["epochs"])
                df["label"] = labels[losses_file_paths.index(path)]
                all_data.append(df)
            except Exception as e:
                logging.warning(f"Failed to load {path}: {e}")

        if not all_data:
            logging.warning("No valid loss data to plot")
            return

        # Create figure
        if plot_lr:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            ax2 = None

        # Plot each loss curve
        colors = plt.cm.tab10(range(len(all_data)))
        for idx, df in enumerate(all_data):
            label = df["label"].iloc[0]
            color = colors[idx]

            # Plot average loss
            ax1.plot(
                df["epoch"],
                df["avg_loss"],
                "-",
                linewidth=2,
                label=label,
                color=color,
            )

            # Plot range if requested
            if plot_range:
                ax1.fill_between(
                    df["epoch"],
                    df["min_loss"],
                    df["max_loss"],
                    alpha=0.1,
                    color=color,
                )

            # Mark best epoch
            loss_data = TrainingLossTracker.load_losses_file(losses_file_paths[idx])
            best_epoch = loss_data.get("best_epoch", -1)
            best_loss = loss_data.get("best_loss", float("inf"))
            if best_epoch > 0:
                ax1.plot(
                    best_epoch,
                    best_loss,
                    "o",
                    markersize=8,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=1,
                )

            # Plot learning rate if requested
            if plot_lr and ax2 is not None:
                ax2.plot(
                    df["epoch"],
                    df["learning_rate"],
                    "-",
                    linewidth=2,
                    label=label,
                    color=color,
                )

        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        if plot_lr and ax2 is not None:
            ax2.set_xlabel("Epoch", fontsize=12)
            ax2.set_ylabel("Learning Rate", fontsize=12)
            ax2.set_title(
                "Learning Rate Schedule Comparison", fontsize=14, fontweight="bold"
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
            ax2.set_yscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved comparison plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
