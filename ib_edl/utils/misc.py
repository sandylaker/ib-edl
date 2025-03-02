import logging
import os.path as osp
from copy import deepcopy
from os import PathLike
from typing import Any, Dict, Optional, Union

import mmengine
import numpy as np
import torch
from transformers.trainer_utils import PredictionOutput

from .logging import setup_logger


def probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    probs = torch.clamp(probs, min=1e-10)
    reference_probs = probs[:, -1]
    logits = torch.log(probs) - torch.log(reference_probs.unsqueeze(1))
    return logits


def get_subset_indices(subset_cfg: Dict[str, Any]) -> np.ndarray:
    subset_cfg = deepcopy(subset_cfg)
    selection_type = subset_cfg.pop('type')
    if selection_type == 'range':
        return np.arange(subset_cfg['start'], subset_cfg['end'], subset_cfg.get('step', 1))
    elif selection_type == 'random':
        rng = np.random.default_rng(subset_cfg.get('seed', 42))
        return rng.choice(np.arange(subset_cfg['start'], subset_cfg['end']), subset_cfg['size'], replace=False)
    else:
        raise ValueError(f'Invalid selection type: {selection_type}')


def save_predictions(
        preds: PredictionOutput, file_path: Union[str, PathLike[str]], logger: Optional[logging.Logger] = None) -> None:
    if osp.dirname(file_path) != '':
        mmengine.mkdir_or_exist(osp.dirname(file_path))

    if isinstance(preds.predictions, tuple):
        # PredictionOutput.predictions can be a tuple of (logits, uncertainties)
        logits, uncertainties = preds.predictions
    else:
        logits, uncertainties = preds.predictions, None

    save_dict = {'logits': logits.astype(np.float16), 'labels': preds.label_ids}
    if uncertainties is not None:
        save_dict['uncertainties'] = uncertainties.astype(np.float16)
    np.savez_compressed(file_path, **save_dict)
    logger = logger if logger is not None else setup_logger('ib-edl')
    logger.info(f'Predictions saved to {file_path}')
