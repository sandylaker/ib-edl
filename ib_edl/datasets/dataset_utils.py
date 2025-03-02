from typing import Any, Dict, List, Union

import torch


def qa_dataset_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Union[List[str], torch.Tensor]]:
    prompts: List[str] = []
    labels: List[int] = []
    for sample in batch:
        prompts.append(sample['prompt'])
        labels.append(sample['label'])

    return {'prompts': prompts, 'labels': torch.LongTensor(labels)}
