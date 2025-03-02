from typing import Optional

import torch
import torch.nn as nn

from ..utils import setup_logger
from .builder import LOSSES


@LOSSES.register_module(name='ce')
class CEBayesRiskLoss(nn.Module):

    def forward(self, evidences: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        alphas = evidences + 1.0
        strengths = torch.sum(alphas, dim=-1, keepdim=True)

        loss = torch.sum(labels * (torch.digamma(strengths) - torch.digamma(alphas)), dim=-1)

        return torch.mean(loss)


def smooth_labels(labels: torch.Tensor, smooth: float) -> torch.Tensor:
    num_classes = labels.size(-1)
    return labels * (1.0 - smooth) + (1 - labels) * smooth / (num_classes - 1)


@LOSSES.register_module(name='ss')
class SSBayesRiskLoss(nn.Module):

    def __init__(self, smooth: Optional[float] = None, lambda_info: Optional[float] = None) -> None:
        super().__init__()
        if smooth is not None:
            assert 0 <= smooth <= 1
        self.smooth = smooth
        self.lambda_info = lambda_info
        self._has_logged = False

    def forward(self, evidences: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=-1, keepdim=True)
        probabilities = alphas / strength

        if self.smooth is not None:
            labels = smooth_labels(labels, self.smooth)

        error = (labels - probabilities)**2
        variance = probabilities * (1.0 - probabilities) / (strength + 1.0)

        if self.lambda_info is not None and self.lambda_info >= 0:
            if not self._has_logged:
                logger = setup_logger('ib-edl')
                logger.info(f'SSBayesRiskLoss: using lambda_info: {self.lambda_info}')
                self._has_logged = True

            # Weighted MSE
            tri_gammas = torch.polygamma(1, alphas)
            error = error * tri_gammas
            variance = variance * tri_gammas
            loss = torch.sum(error + variance, dim=-1)
            # Information penalty
            sum_log_tri_gammas = torch.sum(torch.log(tri_gammas), dim=-1)
            tri_gamma_strength = torch.polygamma(1, strength)
            sum_tri_gamma_ratios = torch.sum(tri_gamma_strength / tri_gammas, dim=-1)
            log_one_minus_sum_ratios = torch.log(1.0 - sum_tri_gamma_ratios)
            info_penalty = sum_log_tri_gammas + log_one_minus_sum_ratios
            loss = loss - self.lambda_info * info_penalty
        else:
            loss = torch.sum(error + variance, dim=-1)

        return torch.mean(loss)


@LOSSES.register_module(name='kl')
class KLDivergenceLoss(nn.Module):

    def __init__(self, modify_alphas: bool = True) -> None:
        super().__init__()
        self.modify_alphas = modify_alphas

    def forward(self, evidences: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_classes = evidences.size(-1)
        alphas = evidences + 1.0
        if self.modify_alphas:
            alphas_tilde = labels + (1.0 - labels) * alphas
        else:
            alphas_tilde = alphas
        strength_tilde = torch.sum(alphas_tilde, dim=-1, keepdim=True)

        # lgamma is the log of the gamma function
        first_term = (
            torch.lgamma(strength_tilde) - torch.lgamma(evidences.new_tensor(num_classes, dtype=torch.float32)) -
            torch.sum(torch.lgamma(alphas_tilde), dim=-1, keepdim=True))
        second_term = torch.sum(
            (alphas_tilde - 1.0) * (torch.digamma(alphas_tilde) - torch.digamma(strength_tilde)), dim=-1, keepdim=True)
        loss = torch.mean(first_term + second_term)

        return loss
