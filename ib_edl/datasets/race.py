from typing import Any, Callable, Dict, Optional

from datasets import load_dataset
from transformers import PreTrainedTokenizer

from .builder import DATASETS
from .classification import ClassificationDataset
from .dataset_utils import qa_dataset_collate_fn


@DATASETS.register_module()
class RaceDataset(ClassificationDataset):

    preamble = (
        'Read the article and return the label of the correct answer. Questions may end with "?" '
        'or require filling in a placeholder "_".\n'
        '\n'
        'Article: {article}\n'
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
        data_set = load_dataset('ehovy/race', 'middle', **dataset_cfg)
        if subset_size is not None:
            data_set = data_set.select(range(subset_size))
        data_set.set_format('torch')
        super().__init__(
            data_set,
            tokenizer,
            n_labels=4,
            preamble=self.preamble,
            add_space=add_space,
            numerical=False,
            noisy_level=noisy_level)

    def _format_prompt(self, sample: Dict[str, Any]) -> str:
        option_labels = ('A', 'B', 'C', 'D')
        choices = '\n'.join([f'{label}) {text}' for text, label in zip(sample['options'], option_labels)])
        prompt = self.preamble.format(article=sample['article'], question=sample['question'], choices=choices)
        return prompt

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        prompt = self._format_prompt(sample)
        label = ord(sample['answer']) - ord('A')
        if self.label_ptb is not None:
            label = self.label_ptb.process_label(index, label)
        return {'prompt': prompt, 'label': label}

    def get_collate_fn(self) -> Callable:
        return qa_dataset_collate_fn
