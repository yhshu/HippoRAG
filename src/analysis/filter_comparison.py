import argparse
import json
import csv
from collections import defaultdict


def mark_gold_passages(facts, gold_passages):
    # Function to append 'in gold passages' or 'not in gold passages' to each fact
    for fact in facts:
        if fact in gold_passages:
            fact.append('in gold passages')
        else:
            fact.append('not in gold passages')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log1', type=str, required=True, help='Path to the first log file')
    parser.add_argument('--log2', type=str, required=True, help='Path to the second log file')
    args = parser.parse_args()

    # Load the logs from the JSON files
    with open(args.log1, 'r') as f:
        log1 = json.load(f)
    with open(args.log2, 'r') as f:
        log2 = json.load(f)

    # Initialize the win, lose, and tie lists
    win = []
    lose = []
    tie = []
    win_metrics = defaultdict(int)
    lose_metrics = defaultdict(int)
    tie_metrics = defaultdict(int)

    # Iterate over both logs and compare ndcg@10 values
    for sample1, sample2 in zip(log1, log2):
        # Extract ndcg@10 values from both logs
        ndcg1 = sample1.get('ndcg@10', 0)
        ndcg2 = sample2.get('ndcg@10', 0)

        # Create a dictionary to store comparison results for each sample
        comparison_result = {
            'query': sample1.get('query'),
            'gold_passages': sample1.get('gold_passages'),
            'log1_ndcg': ndcg1,
            'log2_ndcg': ndcg2,
            'triples_in_supporting_passage': sample1.get('triples_in_supporting_passage'),
            'log1_before_rerank': sample1.get('rerank', {}).get('facts_before_rerank', []),
            'log2_before_rerank': sample2.get('rerank', {}).get('facts_before_rerank', []),
            'log1_after_rerank': sample1.get('rerank', {}).get('facts_after_rerank', []),
            'log2_after_rerank': sample2.get('rerank', {}).get('facts_after_rerank', [])
        }

        # Use the function to mark if facts are in gold passages
        mark_gold_passages(comparison_result['log1_before_rerank'], comparison_result['triples_in_supporting_passage'])
        mark_gold_passages(comparison_result['log2_before_rerank'], comparison_result['triples_in_supporting_passage'])
        mark_gold_passages(comparison_result['log1_after_rerank'], comparison_result['triples_in_supporting_passage'])
        mark_gold_passages(comparison_result['log2_after_rerank'], comparison_result['triples_in_supporting_passage'])

        metrics = {'len_log1_after_rerank': len(comparison_result['log1_after_rerank']), 'len_log2_after_rerank': len(comparison_result['log2_after_rerank']),
                   'num_facts_in_gold_before_rerank': 0, 'num_facts_in_gold_log1_after_rerank': 0, 'num_facts_in_gold_log2_after_rerank': 0, 'empty_log1_after_rerank': 0,
                   'empty_log2_after_rerank': 0}
        for t in comparison_result['log1_before_rerank']:
            if 'in gold passages' in t:
                metrics['num_facts_in_gold_before_rerank'] += 1
        for t in comparison_result['log1_after_rerank']:
            if 'in gold passages' in t:
                metrics['num_facts_in_gold_log1_after_rerank'] += 1
        for t in comparison_result['log2_after_rerank']:
            if 'in gold passages' in t:
                metrics['num_facts_in_gold_log2_after_rerank'] += 1
        if len(comparison_result['log1_after_rerank']) == 0:
            metrics['empty_log1_after_rerank'] += 1
        if len(comparison_result['log2_after_rerank']) == 0:
            metrics['empty_log2_after_rerank'] += 1

        # Compare ndcg@10 values and classify into win, lose, or tie
        if ndcg1 > ndcg2:
            win.append(comparison_result)
            for key, value in metrics.items():
                win_metrics[key] += value
        elif ndcg1 < ndcg2:
            lose.append(comparison_result)
            for key, value in metrics.items():
                lose_metrics[key] += value
        else:
            tie.append(comparison_result)
            for key, value in metrics.items():
                tie_metrics[key] += value


    # Calculate average metrics for win, lose, and tie
    def calculate_average_metrics(total_metrics, count):
        if count == 0:
            return {key: 0 for key in total_metrics}
        return {key: value / count for key, value in total_metrics.items()}


    avg_win_metrics = calculate_average_metrics(win_metrics, len(win))
    avg_lose_metrics = calculate_average_metrics(lose_metrics, len(lose))
    avg_tie_metrics = calculate_average_metrics(tie_metrics, len(tie))
    # Calculate overall metrics
    overall_metrics = defaultdict(int)
    for key in win_metrics:
        overall_metrics[key] = win_metrics[key] + lose_metrics[key] + tie_metrics[key]
    avg_overall_metrics = calculate_average_metrics(overall_metrics, len(win) + len(lose) + len(tie))

    # Print out the results
    print(f'Number of wins: {len(win)}')
    print(f'Number of losses: {len(lose)}')
    print(f'Number of ties: {len(tie)}')
    print(f'Number of total samples: {len(win) + len(lose) + len(tie)}')
    print()

    print('Average metrics for wins:', avg_win_metrics)
    print()
    print('Average metrics for losses:', avg_lose_metrics)
    print()
    print('Average metrics for ties:', avg_tie_metrics)
    print()
    print('Average metrics overall:', avg_overall_metrics)


    # Optionally, write the results to TSV files
    def write_to_tsv(filename, data):
        with open(filename, 'w', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['query', 'ndcg_1', 'ndcg_2', 'facts_before_rerank1', 'facts_after_rerank1', 'facts_before_rerank2', 'facts_after_rerank2', 'gold_passages',
                                 'triples_in_supporting_passage'])
            for item in data:
                tsv_writer.writerow([
                    item['query'],
                    item['log1_ndcg'],
                    item['log2_ndcg'],
                    item['log1_before_rerank'],
                    item['log1_after_rerank'],
                    item['log2_before_rerank'],
                    item['log2_after_rerank'],
                    item['gold_passages'],
                    item['triples_in_supporting_passage']
                ])


    write_to_tsv('win.tsv', win)
    write_to_tsv('lose.tsv', lose)
    write_to_tsv('tie.tsv', tie)
