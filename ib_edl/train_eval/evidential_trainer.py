from typing import Any, Dict, Optional, Tuple, Union

import mmengine
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..utils import setup_logger
from .builder import LOSSES


class EvidentialTrainer(Trainer):

    def __init__(self, cfg: mmengine.Config, target_ids: torch.LongTensor, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.target_ids = target_ids.to(self.args.device)

        edl_loss_cfg = cfg.edl_loss_cfg
        self.bayesian_loss = LOSSES.build(edl_loss_cfg['bayesian_loss'])
        self.reg_loss = LOSSES.build(edl_loss_cfg['reg_loss'])
        self.custom_logger = setup_logger('ib-edl')
        # flag to avoid multiple warnings
        self._warned_reg_weight = False

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[Tensor, Tuple[Tensor, Union[Dict[str, Tensor], Tuple[Tensor, ...]]]]:
        prompts = inputs.pop('prompts')
        labels = inputs.pop('labels').to(self.args.device)
        # labels need to be one-hot encoded
        labels = F.one_hot(labels, num_classes=self.target_ids.size(-1))
        inputs = self.processing_class(prompts, **self.cfg.tokenizer_run_cfg).to(self.args.device)

        loss = torch.tensor(0.0, device=self.args.device)
        sigma = None
        if self.cfg.get('vib', None) is None:
            pre_evidence = model(**inputs).logits[:, -1, self.target_ids]
            # Important: apply softplus to ensure that the evidence is positive
            evidence = F.softplus(pre_evidence)
        else:
            # apply the variational information bottleneck
            evidence, info_loss, sigma = self.apply_ib(model(**inputs).logits, return_sigma=True)
            loss += info_loss
            # labels need to be repeated for each noisy sample, and reshape to (batch_size * num_noises, num_classes)
            labels = labels.unsqueeze(1).repeat(1, self.cfg['vib']['num_noises'], 1).reshape(-1, labels.size(-1))

        labels = labels.to(evidence.dtype)
        # When Trainer.predict is called, the UpdateRegWeightCallback is not called. So self.state has no attribute
        # reg_weight. To overcome this, we retrieve reg_weight from the config.
        if hasattr(self.state, 'reg_weight'):
            reg_weight = self.state.reg_weight
        else:
            reg_weight = self.cfg.edl_loss_cfg['reg_weight_cfg']['final_reg_weight']
            if not self._warned_reg_weight:
                self.custom_logger.warning(
                    f'Trainer state has no attribute reg_weight. '
                    f'Using final_reg_weight: {reg_weight} from config.')
                self._warned_reg_weight = True

        # compute the loss
        loss += self.bayesian_loss(evidence, labels) + reg_weight * self.reg_loss(evidence, labels)

        if return_outputs:
            alphas = evidence + 1.0
            if self.cfg.get('vib', None) is not None:
                # Compute the average of alphas when IB is applied
                alphas = alphas.reshape(len(prompts), self.cfg['vib']['num_noises'], self.target_ids.size(-1))
                alphas = alphas.mean(dim=1)
                assert sigma is not None, 'sigma should not be None when VIB is applied'
                # The Post-hoc adjustment technique mentioned in the Appendix of the paper. We reduce the alphas by
                # sigma_mult * sigma, where sigma_mult is a hyperparameter.
                alphas = torch.clamp(alphas - self.cfg['vib'].get('sigma_mult', 0.0) * sigma, min=1.0)
            strength = torch.sum(alphas, dim=-1, keepdim=True)
            probs = alphas / strength
            # convert probabilities to logits
            probs = torch.clamp(probs, min=1e-10)
            reference_probs = probs[:, -1]
            logits = torch.log(probs) - torch.log(reference_probs.unsqueeze(1))

            num_classes = logits.size(-1)
            uncertainties = num_classes / strength.squeeze(-1)
            # create a dict to mimic the output of most HF models, in order to be compatible with the Trainer API.
            outputs = {'loss': loss, 'logits': logits, 'uncertainties': uncertainties}
            return loss, outputs
        else:
            return loss

    def apply_ib(self, logits: torch.Tensor,
                 return_sigma: bool) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_noises = self.cfg['vib']['num_noises']
        pre_evidences = logits[:, -1, self.target_ids]
        num_classes = pre_evidences.size(-1)
        # Use the last num_classes predicted logits at the last sequence position as std.
        std = F.softplus(logits[:, -1, -num_classes:])
        # repeat std with num_noise times: (batch_size, num_noises, num_classes)
        noise = torch.randn(logits.size(0), num_noises, num_classes, device=logits.device) * std.unsqueeze(1)
        # add noise to evidences: (batch_size, num_noises, num_classes)
        noisy_evidences = pre_evidences.unsqueeze(1) + noise
        # apply softplus to ensure that the evidence is positive
        evidences = F.softplus(noisy_evidences).reshape(logits.shape[0] * num_noises, num_classes)

        # Compute the KL information loss
        info_loss = 0.5 * torch.sum(pre_evidences.pow(2) + std.pow(2) - 2 * std.log() - 1., dim=-1).mean()
        info_loss = info_loss * self.cfg['vib']['beta']
        if return_sigma:
            return evidences, info_loss, std
        else:
            return evidences, info_loss, None


class UpdateRegWeightCallback(TrainerCallback):

    def __init__(self, start_epoch: 0.25, final_reg_weight: float = 1.0) -> None:
        super().__init__()
        self.start_epoch = start_epoch
        self.final_reg_weight = final_reg_weight

    def on_train_begin(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        state.reg_weight = 0.0
        logger = setup_logger('ib-edl')
        logger.info(f'EvidentialTrainer: initialized reg_weight to {state.reg_weight}')

    def on_epoch_begin(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        if self.start_epoch < 1.0:
            start_epoch = int(self.start_epoch * args.num_train_epochs)
        else:
            start_epoch = self.start_epoch
        if state.epoch >= start_epoch:
            state.reg_weight = min(1.0, state.epoch / args.num_train_epochs) * self.final_reg_weight
        else:
            state.reg_weight = 0.0
        logger = setup_logger('ib-edl')
        logger.info(f'Epoch [{state.epoch}] - EvidentialTrainer: updated reg_weight to {state.reg_weight:.4f}')
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.log({'reg_weight': state.reg_weight, 'step': state.global_step, 'epoch': state.epoch})
