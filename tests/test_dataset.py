import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ib_edl.datasets import DATASETS


def test_arc_dataset():
    dataset_dict = {
        'type': 'ARCDataset',
        'dataset_cfg': {
            'split': 'train'
        },
        'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'),
        'name_suffix': 'C',
    }

    dataset = DATASETS.build(dataset_dict)
    assert dataset.target_ids.shape == (5, )
    # training set contains 1119 samples
    assert len(dataset) == 1119
    assert 'prompt' in dataset[0]
    assert 'label' in dataset[0]
    assert isinstance(dataset[0]['prompt'], str)
    # GT answers of the first five samples: [A, B, B, D, B]
    gt_labels = [0, 1, 1, 3, 1]
    for i in range(5):
        assert dataset[i]['label'] == gt_labels[i]

    # test collate_fn
    collate_fn = dataset.get_collate_fn()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        assert 'prompts' in batch
        assert 'labels' in batch
        assert len(batch['prompts']) == 2
        assert torch.all(batch['labels'] == torch.tensor([0, 1]))
        break


def test_obqa_dataset():
    dataset_dict = {
        'type': 'OBQADataset',
        'dataset_cfg': {
            'split': 'train'
        },
        'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'),
    }

    dataset = DATASETS.build(dataset_dict)
    assert dataset.target_ids.shape == (4, )
    assert len(dataset) == 4957
    assert 'prompt' in dataset[0] and len(dataset[0]['prompt']) > 0
    assert 'label' in dataset[0]
    assert isinstance(dataset[0]['prompt'], str)
    # GT answers of the first five samples: [D, D, C, C, D]
    gt_labels = [3, 3, 2, 2, 3]
    for i in range(5):
        assert dataset[i]['label'] == gt_labels[i]

    # test collate_fn
    collate_fn = dataset.get_collate_fn()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        assert 'prompts' in batch
        assert 'labels' in batch
        assert len(batch['prompts']) == 2
        assert torch.all(batch['labels'] == torch.tensor([3, 3]))
        break


def test_csqa_dataset():
    dataset_dict = {
        'type': 'CSQADataset',
        'dataset_cfg': {
            'split': 'train'
        },
        'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'),
    }

    dataset = DATASETS.build(dataset_dict)
    assert dataset.target_ids.shape == (5, )
    assert len(dataset) == 9741
    assert 'prompt' in dataset[0] and len(dataset[0]['prompt']) > 0
    assert 'label' in dataset[0]
    assert isinstance(dataset[0]['prompt'], str)
    # GT answers of the first five samples: [A, B, A, D, C]
    gt_labels = [0, 1, 0, 3, 2]
    for i in range(5):
        assert dataset[i]['label'] == gt_labels[i]

    # test collate_fn
    collate_fn = dataset.get_collate_fn()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        assert 'prompts' in batch
        assert 'labels' in batch
        assert len(batch['prompts']) == 2
        assert torch.all(batch['labels'] == torch.tensor([0, 1]))
        break


def test_sciq_dataset():
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset_dict = {
        'type': 'SciQDataset',
        'dataset_cfg': {
            'split': 'train'
        },
        'tokenizer': tokenizer,
    }

    dataset = DATASETS.build(dataset_dict)
    assert dataset.target_ids.shape == (4, )
    assert len(dataset) == 11679
    assert 'prompt' in dataset[0] and len(dataset[0]['prompt']) > 0
    assert 'label' in dataset[0]
    assert isinstance(dataset[0]['prompt'], str)
    # GT answers of the first five samples: [A, D, C, B, B]
    gt_labels = [0, 3, 2, 1, 1]
    for i in range(5):
        assert dataset[i]['label'] == gt_labels[i]

    # test collate_fn
    collate_fn = dataset.get_collate_fn()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        assert 'prompts' in batch
        assert 'labels' in batch
        assert len(batch['prompts']) == 2
        assert torch.all(batch['labels'] == torch.tensor([0, 3]))
        break


def test_race_dataset():
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset_dict = {
        'type': 'RaceDataset',
        'dataset_cfg': {
            'split': 'train'
        },
        'tokenizer': tokenizer,
    }

    dataset = DATASETS.build(dataset_dict)
    assert dataset.target_ids.shape == (4, )
    assert len(dataset) == 25421
    assert 'prompt' in dataset[0] and len(dataset[0]['prompt']) > 0
    assert 'label' in dataset[0]
    assert isinstance(dataset[0]['prompt'], str)
    # GT answers of the first five samples: [B, C, A, C, C]
    gt_labels = [1, 2, 0, 2, 2]
    for i in range(5):
        assert dataset[i]['label'] == gt_labels[i]

    # test collate_fn
    collate_fn = dataset.get_collate_fn()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        assert 'prompts' in batch
        assert 'labels' in batch
        assert len(batch['prompts']) == 2
        assert torch.all(batch['labels'] == torch.tensor([1, 2]))
        break


def test_mmlu_dataset():
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset_dict = {
        'type': 'MMLUDataset',
        'task': 'cs_area',
        'dataset_cfg': {
            'split': 'test'
        },
        'tokenizer': tokenizer,
    }
    dataset = DATASETS.build(dataset_dict)
    assert dataset.target_ids.shape == (4, )
    assert len(dataset) == 412
    assert 'prompt' in dataset[0] and len(dataset[0]['prompt']) > 0
    assert 'label' in dataset[0]
    assert isinstance(dataset[0]['prompt'], str)
    # GT answers of the first five samples: [A, C, B, A, C]
    gt_labels = [0, 2, 1, 0, 2]
    for i in range(5):
        assert dataset[i]['label'] == gt_labels[i]

    # test collate_fn
    collate_fn = dataset.get_collate_fn()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        assert 'prompts' in batch
        assert 'labels' in batch
        assert len(batch['prompts']) == 2
        assert torch.all(batch['labels'] == torch.tensor([0, 2]))
        break
