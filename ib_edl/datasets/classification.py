from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ..utils import setup_logger


class ClassificationDataset(Dataset):
    """
    An abstract base dataset for sequence classification problems. Multiple
    choice QA problems could also be made a subclass of this class with an
    appropriate collation / formatting.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        n_labels: int,
        preamble: str,
        add_space: bool = False,
        numerical: bool = False,
        boolean: bool = False,
        noisy_level: Optional[float] = None,
    ):
        """
        Args:
            dataset: The loaded Dataset
            tokenizer: The model tokenizer
            n_labels: The number of labels / gt_classes for each question
            preamble: Preamble for general pre-trained / 'CausalLM' models
            add_space: Add an explicit space suffix between preamble and answer tokens.
            numerical: whether labels are numerical (0, 1, ...) or alphabetical (A, B, ...)
        """
        self.dataset = dataset
        self.n_labels = n_labels
        self.preamble = preamble
        self.add_space = add_space
        self.tokenizer = tokenizer
        self.numerical = numerical

        spc = ' ' if self.add_space else ''
        """Token ids of class labels. Example [345, 673, 736]."""
        if numerical and boolean:
            raise ValueError('Question type cannot be both numerical and boolean')
        if boolean:
            labels = [f'{spc}True', f'{spc}False']
        elif numerical:
            labels = [f'{spc}{i}' for i in range(self.n_labels)]
        else:  # alphabetical
            labels = [f"{spc}{chr(ord('A')+i)}" for i in range(self.n_labels)]
        # assume these encode to single tokens
        self._target_ids = tokenizer(labels, return_tensors='pt', add_special_tokens=False).input_ids[:, -1:]
        self._target_ids = self._target_ids.squeeze()
        """A mapping from label _indices_ to target token ids. This is only useful for CausalLM models.
        Example: {(0, 345), (1, 673), (2, 736)}
        """
        self.label2target = OrderedDict([(i, self.target_ids[i]) for i in range(n_labels)])
        # misnomer: should be target 2 label _index_
        self.target2label = OrderedDict([(self.target_ids[i], i) for i in range(n_labels)])

        if noisy_level is not None and noisy_level > 0:
            logger = setup_logger('ib-edl')
            logger.info(f'Dataset: will apply label perturbation with noisy level {noisy_level}')
            self.label_ptb = LabelPerturbation(len(self.dataset), n_labels, noisy_level)
        else:
            self.label_ptb: Optional[LabelPerturbation] = None

    @property
    def target_ids(self):
        return self._target_ids.clone().detach()

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError


class LabelPerturbation:

    def __init__(self, num_samples: int, num_classes: int, noisy_level: float) -> None:
        self.noisy_level = noisy_level
        self.num_samples = num_samples
        self.num_classes = num_classes

        # 1 stands for perturbation, 0 stands for no perturbation
        apply_ptb = np.zeros(num_samples, dtype=bool)
        # apply perturbation to noisy_level * num_samples
        apply_ptb[:int(noisy_level * num_samples)] = True
        self._rng = np.random.default_rng(42)
        self._rng.shuffle(apply_ptb)

        self.apply_ptb = apply_ptb

    def process_label(self, sample_index: int, label: int) -> int:
        if self.apply_ptb[sample_index]:
            # perturb the label to a random class except the original one
            choices = np.arange(self.num_classes)
            choices = np.delete(choices, label)
            return self._rng.choice(choices)
        else:
            return label
