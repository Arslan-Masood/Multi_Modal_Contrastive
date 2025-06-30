
import json
import os
import shutil
from typing import Dict, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from dataset import _split_data
from utils import utils

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from neptune.types import File
from pytorch_lightning.utilities.seed import seed_everything



def build_dataloaders(
    dataset: Dataset,
    batch_size: int,
    splits: Dict[str, str] = None,
    **kwargs,
) -> Dict[str, DataLoader]:
    split_dataset = _split_data(dataset, splits=splits)
    dataloaders = {}
    
    # Get collate_fn from dataset if it exists
    collate_fn = getattr(dataset, 'collate_fn', None)
    
    print("\nDataloader Sizes:")
    print("-" * 50)

    for split, ds in split_dataset.items():
        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=split in ["train"],
            drop_last=split in ["train"],
            collate_fn=collate_fn,
            **kwargs,
        )

        # Print dataset and dataloader lengths
        num_samples = len(ds)
        num_batches = len(dataloaders[split])
        print(f"{split.upper()}:")
        print(f"  • Number of samples: {num_samples}")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Number of batches: {num_batches}")
        print(f"  • Drop last: {split in ['train']}")
        print("-" * 50)
    return dataloaders


def train(cfg: Union[Dict, DictConfig]) -> nn.Module:
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, workers=True)

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    model.set_optimizer(optimizer)

    if hasattr(cfg, "scheduler"):
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        model.set_scheduler(scheduler, OmegaConf.to_container(cfg.scheduler_config))

    dataloaders = hydra.utils.call(cfg.dataloaders)
    mock_inputs = iter(dataloaders["train"]).next()
    _ = model(**mock_inputs["inputs"])

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.logger.experiment["model/hyper-parameters"] = (OmegaConf.to_container(cfg))
    print("Hyperparameters logged to Neptune.")
    trainer.fit(model, dataloaders["train"], dataloaders["val"])


    checkpoint_dir = trainer.checkpoint_callback.dirpath
    with open(os.path.join(checkpoint_dir, "config.yml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    best = {
        "best_ckpt_path": trainer.checkpoint_callback.best_model_path,
        "best_metric": trainer.early_stopping_callback.best_score.item(),
    }

    best_str = json.dumps(best, indent=4)
    print(best_str)

    with open(os.path.join(checkpoint_dir, "best_ckpt.json"), "w") as f:
        f.write(best_str)

    shutil.copyfile(
        best["best_ckpt_path"],
        os.path.join(checkpoint_dir, "best_ckpt.ckpt")
    )
    return best



class TrainingMonitor(pl.Callback):
    """
    A callback to generate and log plots during training for model monitoring.

    This callback generates a heatmap of the contrastive similarity matrix (logits)
    at the end of each training epoch.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the training epoch.
        Generates and logs plots.
        """
        # Ensure the logger is available
        if trainer.logger is None:
            return

        # 1. Get the stored sample batch from the LightningModule
        sample_batch = pl_module.sample_train_batch
        if sample_batch is None:
            return

        # 2. Perform a forward pass to get the logits
        pl_module.eval() # Set model to evaluation mode
        with torch.no_grad():
            logits_ab, _, _ = pl_module.forward(**sample_batch["inputs"])
            temperature = pl_module.temperature
            scaled_logits_ab = logits_ab * temperature
        pl_module.train() # Set model back to training mode
        
        # Apply softmax along the dimension that represents the "keys" or candidates
        # For a similarity matrix (query x key), typically dim=1 is for keys.
        probabilities_ab = F.softmax(scaled_logits_ab, dim=1) 
        # --------------------------------------------------------

        # 3. Plot the Morphological Probabilities
        self.plot_probabilities_heatmap(
            trainer=trainer, 
            probabilities=probabilities_ab.cpu().numpy(), # Pass probabilities
            title="Morphological Similarity-Training (Softmax Probabilities a-b)" # Update title
        )

        print("Logits shape:", logits_ab.shape)
        print("Logits stats:", logits_ab.min().item(), logits_ab.max().item(), logits_ab.mean().item())
        print("Logits example (first row):", logits_ab[0][:5])

        probabilities_ab = F.softmax(logits_ab, dim=1)
        print("Probabilities example (first row):", probabilities_ab[0][:5])

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the training epoch.
        Generates and logs plots.
        """
        # Ensure the logger is available
        if trainer.logger is None:
            return

        # 1. Get the stored sample batch from the LightningModule
        sample_batch = pl_module.sample_val_batch
        if sample_batch is None:
            return

        # 2. Perform a forward pass to get the logits
        pl_module.eval() # Set model to evaluation mode
        with torch.no_grad():
            logits_ab, _, _ = pl_module.forward(**sample_batch["inputs"])
            temperature = pl_module.temperature
            scaled_logits_ab = logits_ab * temperature
        pl_module.train() # Set model back to training mode
        
        # Apply softmax along the dimension that represents the "keys" or candidates
        # For a similarity matrix (query x key), typically dim=1 is for keys.
        probabilities_ab = F.softmax(scaled_logits_ab, dim=1) 
        # --------------------------------------------------------

        # 3. Plot the Morphological Probabilities
        self.plot_probabilities_heatmap(
            trainer=trainer, 
            probabilities=probabilities_ab.cpu().numpy(), # Pass probabilities
            title="Morphological Similarity-Validation (Softmax Probabilities a-b)" # Update title
        )

    def plot_probabilities_heatmap(self, trainer, probabilities, title): # Renamed method
        """
        Generates and logs a heatmap of the logits matrix.
        
        The diagonal represents the similarity scores of positive pairs, which should be high.
        The off-diagonal represents negative pairs, which should be low.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(probabilities, annot=False, cmap="viridis", ax=ax, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        
        # 4. Log the figure to the logger (e.g., TensorBoard)
        trainer.logger.experiment[f"training_plots/{title}"].append(fig)
        plt.close(fig) # Close the figure to free up memory
