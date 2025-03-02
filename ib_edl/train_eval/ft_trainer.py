from typing import Any, Dict, Optional, Tuple, Union

import mmengine
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Trainer


class FTTrainer(Trainer):

    def __init__(self, cfg: mmengine.Config, target_ids: torch.LongTensor, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.target_ids = target_ids.to(self.args.device)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[Tensor, Tuple[Tensor, Union[Dict[str, Tensor], Tuple[Tensor, ...]]]]:
        prompts = inputs.pop('prompts')
        labels = inputs.pop('labels').to(self.args.device)
        inputs = self.processing_class(prompts, **self.cfg.tokenizer_run_cfg).to(self.args.device)
        logits = model(**inputs).logits[:, -1, self.target_ids]
        loss = F.cross_entropy(logits, labels)
        # create a dict to mimic the output of most huggingface models, in order to be compatible with the Trainer API.
        outputs = {'loss': loss, 'logits': logits}

        return (loss, outputs) if return_outputs else loss
