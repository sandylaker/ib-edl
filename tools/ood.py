import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import numpy as np
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score

from ib_edl.utils import setup_logger


def parse_args():
    parser = ArgumentParser('Out-of-distribution detection.')
    parser.add_argument('id_preds', help='Path to the predictions of the in-distribution data.')
    parser.add_argument('ood_preds', help='Path to the predictions of the out-of-distribution data.')
    parser.add_argument('--work-dir', '-w', default='workdirs/debug/', help='Working directory.')
    parser.add_argument(
        '--log-file', '-f', help='Log file name without extension. If not specified, the timestamp will be used.')

    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)

    log_file_name = datetime.now().strftime('%m%d_%H%M_%S') if args.log_file is None else args.log_file
    logger = setup_logger(
        name='ib-edl',
        filepath=osp.join(work_dir, f'{log_file_name}.log'),
    )
    logger.info(f'Loading predictions from {args.id_preds}')
    id_preds = np.load(osp.join(args.id_preds))
    logger.info(f'Loading predictions from {args.ood_preds}')
    ood_preds = np.load(osp.join(args.ood_preds))

    num_id_samples = len(id_preds['labels'])
    num_ood_samples = len(ood_preds['labels'])
    det_labels = np.concatenate([np.ones(num_id_samples, dtype=np.int64), np.zeros(num_ood_samples, np.int64)], axis=0)

    # Detection using max probability
    id_probabilities = softmax(id_preds['logits'], axis=-1)
    ood_probabilities = softmax(ood_preds['logits'], axis=-1)
    mp_id_scores = np.max(id_probabilities, axis=1)
    mp_ood_scores = np.max(ood_probabilities, axis=1)
    mp_det_scores = np.concatenate([mp_id_scores, mp_ood_scores], axis=0)
    mp_au_roc = roc_auc_score(det_labels, mp_det_scores)
    mp_au_pr = average_precision_score(det_labels, mp_det_scores)
    logger.info(f'Using MP: AUROC: {mp_au_roc:.4f}, AUPR: {mp_au_pr:.4f}')

    # Detection using uncertainty mass. Only the EDL models have UM as uncertainties.
    # Methods like Ensemble do not have key 'uncertainties' in the predictions.
    if 'uncertainties' in id_preds and 'uncertainties' in ood_preds:
        try:
            um_id_scores = 1 / np.clip(id_preds['uncertainties'].astype(np.float64), a_min=1e-8, a_max=None)
            um_ood_scores = 1 / np.clip(ood_preds['uncertainties'].astype(np.float64), a_min=1e-8, a_max=None)
            um_det_scores = np.concatenate([um_id_scores, um_ood_scores], axis=0)
            um_au_roc = roc_auc_score(det_labels, um_det_scores)
            um_au_pr = average_precision_score(det_labels, um_det_scores)
            logger.info(f'Using UM: AUROC: {um_au_roc:.4f}, AUPR: {um_au_pr:.4f}')
        except ValueError as e:
            logger.warning(f'"uncertainties" are in the predictions but could not be loaded. Error: {e}')
    else:
        logger.info('"uncertainties" are not in the predictions. Skipping OOD detection using UM.')


if __name__ == '__main__':
    main()
