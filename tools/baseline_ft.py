import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import torch
import wandb
from mmengine.runner.utils import set_random_seed
from transformers import TrainingArguments

from ib_edl.datasets import DATASETS
from ib_edl.models import get_model_and_tokenizer
from ib_edl.train_eval import ClassificationMetric, FTTrainer, plot_predictions
from ib_edl.utils import save_predictions, setup_logger


def parse_args():
    parser = ArgumentParser('Fine-tune model.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--work-dir', '-w', help='Working directory.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument('--run-name', '-n', help='Run name of wandb.')
    parser.add_argument('--run-group', '-g', help='Run group of wandb.')
    parser.add_argument('--skip-ft', '-s', action='store_true', help='Skip fine-tuning.')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--cfg-options',
        '-o',
        nargs='+',
        action=mmengine.DictAction,
        help='Override the config entry using xxx=yyy format.')

    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)
    device = torch.device(f'cuda:{args.gpu_id}')

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    timestamp = datetime.now().strftime('%m%d_%H%M_%S')
    logger = setup_logger(
        name='ib-edl',
        filepath=osp.join(work_dir, f'{timestamp}.log'),
    )
    logger.info('Using config:\n' + '=' * 60 + f'\n{cfg.pretty_text}\n' + '=' * 60)
    cfg.dump(osp.join(work_dir, f'{osp.splitext(osp.basename(cfg.filename))[0]}_{timestamp}.yaml'))

    if not args.no_wandb:
        run_name = args.run_name if args.run_name is not None else timestamp
        run_group = args.run_group if args.run_group is not None else None
        wandb.init(project='ib-edl', dir=work_dir, name=run_name, group=run_group)
        wandb.config.update({'ib-edl_config': cfg.to_dict()})

    model, tokenizer = get_model_and_tokenizer(**cfg.model, device=device)

    train_set = DATASETS.build(cfg.data['train'], default_args=dict(tokenizer=tokenizer))
    val_set = DATASETS.build(cfg.data['val'], default_args=dict(tokenizer=tokenizer))
    test_set = DATASETS.build(cfg.data['test'], default_args=dict(tokenizer=tokenizer))
    train_target_ids = train_set.target_ids
    val_target_ids = val_set.target_ids
    test_target_ids = test_set.target_ids
    assert torch.all(train_target_ids == val_target_ids), 'target_ids of train and val sets are different.'

    if type(train_set) is type(test_set):
        target_ids = train_target_ids
    else:
        assert args.skip_ft, ('Train and test sets are of different types, indicating that the experiment is In-Out '
                              'distribution test, where the model is train on one dataset and tested on another. In '
                              'this case, the model should not be fine-tuned.')
        target_ids = test_target_ids

    training_args = TrainingArguments(
        output_dir=work_dir,
        logging_dir=work_dir,
        report_to='wandb' if not args.no_wandb else 'none',
        remove_unused_columns=False,
        run_name=timestamp if args.run_name is None else args.run_name,
        **cfg.train_cfg,
    )

    trainer = FTTrainer(
        cfg=cfg,
        target_ids=target_ids,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=ClassificationMetric(num_classes=target_ids.shape[-1]),
        data_collator=train_set.get_collate_fn(),
    )

    if not args.skip_ft:
        trainer.train()
        logger.info('Fine-tuning finished.')
    logger.info('Start evaluating the model on test set... ')
    test_metrics = trainer.evaluate(eval_dataset=test_set, metric_key_prefix='test')
    for key, value in test_metrics.items():
        logger.info(f'MAP: Test metrics: {key}: {value}')

    if cfg.process_preds['npz_file'] is not None or cfg.process_preds['do_plot']:
        logger.info('Start re-run prediction on test set and saving results')
        predictions = trainer.predict(test_set)
        logger.info('Start processing predictions for MAP...')
        if cfg.process_preds['npz_file'] is not None:
            preds_save_path = osp.join(work_dir, 'preds', cfg.process_preds['npz_file'])
            save_predictions(predictions, preds_save_path, logger=logger)
        if cfg.process_preds['do_plot']:
            plot_predictions(predictions, cfg.process_preds['plot_cfg'], work_dir)


if __name__ == '__main__':
    main()
