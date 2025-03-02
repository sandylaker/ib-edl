from typing import Dict

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, CalibrationError
from transformers import EvalPrediction


class ClassificationMetric:

    def __init__(self, num_classes: int, n_bins: int = 15) -> None:
        self.num_classes = num_classes
        self.acc_metric = Accuracy(task='multiclass', num_classes=num_classes)
        self.ece_metric = CalibrationError(task='multiclass', num_classes=num_classes, n_bins=n_bins)

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        # eval_pred.predictions can be a tuple of (logits, uncertainties)
        if isinstance(eval_pred.predictions, tuple):
            logits = torch.tensor(eval_pred.predictions[0])
        else:
            logits = torch.tensor(eval_pred.predictions)

        labels = torch.tensor(eval_pred.label_ids)

        nll = F.cross_entropy(logits, labels).item()
        self.acc_metric(logits, labels)
        self.ece_metric(logits, labels)

        result = {
            'accuracy': self.acc_metric.compute().item(),
            'ece': self.ece_metric.compute().item(),
            'nll': nll,
        }

        # the logits and labels are of the whole dataset. Therefore, when this method is called,
        # the evaluation is finished for one epoch. We need to reset the metrics.
        self.acc_metric.reset()
        self.ece_metric.reset()

        return result
