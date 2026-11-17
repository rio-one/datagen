"""Read official Omnidata inference weights without installing the Lightning training stack."""
from pathlib import Path

import torch


class _LegacyModelCheckpoint:
    """Placeholder for an unused callback class key in the official training metadata."""


def load_omnidata_checkpoint(path: Path):
    # Official v2 checkpoints include this Lightning class as a callbacks mapping key.
    # Its training implementation is not needed to recover the tensor state_dict.
    legacy_name = "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"
    with torch.serialization.safe_globals([(_LegacyModelCheckpoint, legacy_name)]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected an Omnidata checkpoint dictionary")
    return checkpoint
