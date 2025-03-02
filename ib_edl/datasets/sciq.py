from typing import Any, Callable, Dict, Optional

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from .builder import DATASETS
from .classification import ClassificationDataset
from .dataset_utils import qa_dataset_collate_fn


@DATASETS.register_module()
class SciQDataset(ClassificationDataset):

    preamble = (
        'Return the label of the correct answer for the question below.\n'
        '\n'
        'Question: {question}\n'
        'Choices:\n'
        '{choices}\n'
        'Answer:')

    def __init__(
        self,
        dataset_cfg: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        add_space: bool = True,
        subset_size: Optional[int] = None,
        noisy_level: Optional[float] = None,
    ) -> None:
        data_set = load_dataset('allenai/sciq', **dataset_cfg)
        if subset_size is not None:
            data_set = data_set.select(range(subset_size))
        data_set.set_format('torch')
        rng = np.random.default_rng(42)
        self._gt_labels = rng.integers(0, 4, len(data_set))
        super().__init__(
            data_set,
            tokenizer,
            n_labels=4,
            preamble=self.preamble,
            add_space=add_space,
            numerical=False,
            noisy_level=noisy_level)

    def _format_prompt(self, sample: Dict[str, Any], gt_label: int) -> str:
        options = ('A', 'B', 'C', 'D')
        if gt_label == 0:
            choice_texts = (
                sample['correct_answer'], sample['distractor1'], sample['distractor2'], sample['distractor3'])
        elif gt_label == 1:
            choice_texts = (
                sample['distractor1'], sample['correct_answer'], sample['distractor2'], sample['distractor3'])
        elif gt_label == 2:
            choice_texts = (
                sample['distractor1'], sample['distractor2'], sample['correct_answer'], sample['distractor3'])
        elif gt_label == 3:
            choice_texts = (
                sample['distractor1'], sample['distractor2'], sample['distractor3'], sample['correct_answer'])
        else:
            raise ValueError('Invalid gt_label')
        choices = '\n'.join([f'{label}) {text}' for text, label in zip(choice_texts, options)])
        prompt = self.preamble.format(question=sample['question'], choices=choices)
        return prompt

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        label = self._gt_labels[index]
        prompt = self._format_prompt(sample, label)
        if self.label_ptb is not None:
            label = self.label_ptb.process_label(index, label)
        return {'prompt': prompt, 'label': label}

    def get_collate_fn(self) -> Callable:
        return qa_dataset_collate_fn
