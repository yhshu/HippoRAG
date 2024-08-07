import sys

sys.path.append('.')

import argparse
import json

from tqdm import tqdm

from src.pangu.openkg_pangu_api import PanguForOpenKG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--mid_to_node', type=str)
    parser.add_argument('--node_to_mid', type=str)
    parser.add_argument('--beam_size', type=int, default=5)
    args = parser.parse_args()

    data = json.load(open(f'data/{args.dataset}.json'))
    pangu = PanguForOpenKG(args.llm, mid_to_node_path=args.mid_to_node, node_to_mid_path=args.node_to_mid)

    max_steps = 4 if args.dataset.startswith('musique') else 2
    assert args.beam_size is not None and args.beam_size > 0

    metrics = {'em': 0, 'f1': 0}
    logs = []
    log_output_path = f'output/ttl_kg/{args.dataset}_{args.llm}_qa_log.json'

    for i, sample in tqdm(enumerate(data), total=len(data), desc='Pangu QA'):
        question = sample['question']
        gold_ans = sample['answer']
        predicted_queries, beams = pangu.text_to_query(question, max_steps=max_steps, beam_size=args.beam_size)

        pred_ans = ''
        pred_lisp_repr = ''
        final_step = 0
        for p in predicted_queries:
            if len(p['labels']) == 0:
                continue
            pred_ans = '; '.join(p['labels'])
            pred_lisp_repr = p['s-expression_repr']
            final_step = p['final_step']
            break

        if args.dataset == 'hotpotqa':
            from src.qa.hotpotqa_evaluation import update_answer

            em, f1, precision, recall = update_answer({'em': 0, 'f1': 0, 'precision': 0, 'recall': 0}, pred_ans, gold_ans)
        elif args.dataset == 'musique':
            from src.qa.musique_evaluation import evaluate

            em, f1 = evaluate({'predicted_answer': pred_ans}, sample)
        elif args.dataset == '2wikimultihopqa':
            from src.qa.twowikimultihopqa_evaluation import exact_match_score, f1_score

            em = 1 if exact_match_score(pred_ans, gold_ans) else 0
            f1, precision, recall = f1_score(pred_ans, gold_ans)
        else:
            raise NotImplementedError(f'Unknown dataset {args.dataset}')

        metrics['em'] += em
        metrics['f1'] += f1
        print('[Question]', question)
        print('[Step]', final_step)
        print('[Gold]', gold_ans)
        print('[Prediction]', pred_ans)
        print('[EM]', em, '[F1]', f1)
        print('[Avg EM]', metrics['em'] / (i + 1), '[Avg F1]', metrics['f1'] / (i + 1))
        logs.append({'question': question, 'gold_ans': gold_ans, 'pred_ans': pred_ans,
                     'pred_lisp_repr': pred_lisp_repr, 'final_step': final_step,
                     'em': em, 'f1': f1, 'beams': beams})

        log_output_path = f'output/ttl_kg/{args.dataset}_{args.llm}_qa_log.json'
        json.dump(logs, open(log_output_path, 'w'), indent=4)
    print(f'QA logs saved to {log_output_path}')
