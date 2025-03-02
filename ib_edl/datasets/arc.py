from typing import Any, Callable, Dict, Optional

from datasets import load_dataset
from transformers import PreTrainedTokenizer

from .builder import DATASETS
from .classification import ClassificationDataset
from .dataset_utils import qa_dataset_collate_fn


@DATASETS.register_module()
class ARCDataset(ClassificationDataset):

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
        name_suffix: str = 'C',
        add_space: bool = True,
        subset_size: Optional[int] = None,
        noisy_level: Optional[float] = None,
    ):
        if name_suffix == 'C':
            arc_name = 'ARC-Challenge'
        elif name_suffix == 'E':
            arc_name = 'ARC-Easy'
        else:
            raise ValueError(f'Invalid ARC name: {name_suffix}')
        data_set = load_dataset('ai2_arc', arc_name, **dataset_cfg)
        if subset_size is not None:
            data_set = data_set.select(range(subset_size))
        data_set.set_format('torch')
        super().__init__(
            data_set,
            tokenizer,
            n_labels=5,
            preamble=self.preamble,
            add_space=add_space,
            numerical=False,
            noisy_level=noisy_level)

    def _format_prompt(self, sample: Dict[str, Any]) -> str:
        choices = '\n'.join(
            [f'{label}) {text}' for text, label in zip(sample['choices']['text'], sample['choices']['label'])])
        return self.preamble.format(question=sample['question'], choices=choices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        prompt = self._format_prompt(sample)
        class_alpha = ord(sample['answerKey']) - ord('A')

        try:
            class_num = int(sample['answerKey']) - 1
        except:
            class_num = -1

        label = class_num if class_alpha < 0 else class_alpha
        if self.label_ptb is not None:
            label = self.label_ptb.process_label(index, label)
        return {'prompt': prompt, 'label': label}

    def get_collate_fn(self) -> Callable:
        return qa_dataset_collate_fn
