import sys

sys.path.append('.')

import argparse
import json
from collections import defaultdict

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, help='Path to the log file')
    args = parser.parse_args()

    with open(args.log, 'r') as f:
        log = json.load(f)

    metrics = defaultdict(float)
    for sample in tqdm(log):
        assert 'rerank' in sample
        num_before_filter = len(sample['rerank']['facts_before_rerank'])
        num_after_filter = len(sample['rerank']['facts_after_rerank'])
        metrics['num_before_filter'] += num_before_filter
        metrics['num_after_filter'] += num_after_filter
        if num_after_filter == 0:
            metrics['empty_after_filter'] += 1

    metrics['empty_after_filter'] /= len(log)
    metrics['num_before_filter'] /= len(log)
    metrics['num_after_filter'] /= len(log)
    for key, value in metrics.items():
        print(f'{key}: {round(value, 4)}')
