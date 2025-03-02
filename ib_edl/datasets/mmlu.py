from typing import Any, Callable, Dict, Optional

from datasets import concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer

from .builder import DATASETS
from .classification import ClassificationDataset
from .dataset_utils import qa_dataset_collate_fn


@DATASETS.register_module()
class MMLUDataset(ClassificationDataset):

    preamble = (
        'Return the label of the correct answer for the question below.\n'
        '\n'
        'Question: {question}\n'
        'Choices:\n'
        '{choices}\n'
        'Answer:')

    compo_task_dict = {
        'cs_area':
        ['college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning'],
        'eng_area': ['electrical_engineering'],
        'law_area': ['international_law', 'jurisprudence', 'professional_law'],
        'health_area': [
            'anatomy',
            'clinical_knowledge',
            'college_medicine',
            'human_aging',
            'nutrition',
            'professional_medicine',
            'virology'
        ],
        'math_area': ['college_mathematics', 'high_school_mathematics', 'abstract_algebra'],
    }

    def __init__(
        self,
        task: str,
        dataset_cfg: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        add_space: bool = True,
        subset_size: Optional[int] = None,
        noisy_level: Optional[float] = None,
    ) -> None:
        if task in self.compo_task_dict:
            data_set_list = [load_dataset('cais/mmlu', name, **dataset_cfg) for name in self.compo_task_dict[task]]
            data_set = concatenate_datasets(data_set_list)
        else:
            data_set = load_dataset('cais/mmlu', task, **dataset_cfg)
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
        option_labels = ['A', 'B', 'C', 'D']
        choices = '\n'.join([f'{label}) {text}' for label, text in zip(option_labels, sample['choices'])])
        prompt = self.preamble.format(question=sample['question'], choices=choices)
        return prompt

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        prompt = self._format_prompt(sample)
        # sample['answer'] is a tensor e.g. tensor(0).
        label = sample['answer'].item()
        if self.label_ptb is not None:
            label = self.label_ptb.process_label(index, label)
        return {'prompt': prompt, 'label': label}

    def get_collate_fn(self) -> Callable:
        return qa_dataset_collate_fn
