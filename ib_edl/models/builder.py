from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import peft
import torch
import transformers
from peft import get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils import Device, setup_logger


def get_model_and_tokenizer(
        model_name_or_path: str,
        model_cfg: Dict[str, Any],
        tokenizer_cfg: Dict[str, Any],
        special_tokens: Dict[str, str],
        use_peft: bool = False,
        peft_cfg: Optional[Dict[str, Any]] = None,
        peft_path: Optional[str] = None,
        device: Device = 'cuda:0') -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logger = setup_logger('ib-edl')
    model_cfg = deepcopy(model_cfg)
    tokenizer_cfg = deepcopy(tokenizer_cfg)
    peft_cfg = deepcopy(peft_cfg)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    logger.info('Set torch.backends.cuda.enable_flash_sdp and torch.backends.cuda.enable_mem_efficient_sdp to False.')

    model_cls_name = model_cfg.pop('type')
    model_cls = getattr(transformers, model_cls_name)

    model_torch_dtype = model_cfg.get('torch_dtype', None)
    if model_torch_dtype is not None and model_torch_dtype != 'auto':
        model_cfg.update({'torch_dtype': getattr(torch, model_cfg['torch_dtype'])})
    model = model_cls.from_pretrained(model_name_or_path, **model_cfg)

    if peft_path is not None:
        # if lora weights are available
        logger.info(
            'get_model_and_tokenizer: peft_path is not None. So use_peft does not take effect. '
            'PEFT parameters will be loaded.')
        logger.info(f'Loading peft parameters from {peft_path}')
        model = peft.PeftModel.from_pretrained(model, peft_path, is_trainable=True)
    else:
        if peft_cfg is not None and use_peft:
            logger.info('Setting up PEFT.')
            peft_cls_name = peft_cfg.pop('type')
            peft_cls = getattr(peft, peft_cls_name)
            peft_cfg_object: peft.PeftConfig = peft_cls(**peft_cfg)
            model = get_peft_model(model, peft_cfg_object)

    model.to(device)

    tokenizer_cls = getattr(transformers, tokenizer_cfg.pop('type'))
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path, **tokenizer_cfg)
    tokenizer_special_tokens = {
        k: getattr(tokenizer, v.split('.')[-1]) if isinstance(v, str) and v.startswith('tokenizer') else v
        for k,
        v in special_tokens.items()
    }
    if len(tokenizer_special_tokens) > 0:
        tokenizer.add_special_tokens(tokenizer_special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
