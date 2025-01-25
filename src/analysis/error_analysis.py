import argparse
import os

import json
import random
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str,
                        default='output/ircot_retrieval/musique/musique_hipporag_GT_pnu_E_meta-llama_Llama-3_3-70B-Instruct_R_nvidia_NV-Embed-v2_L_nvidia_NV-Embed-v2_query_to_fact_RE_meta-llama_Llama-3_3-70B-Instruct_filter_llama3.3-70B-Instruct_demo_1_step_1_top_10_ppr_damp_0.5_sim_0.8_LT_5.json')
    args = parser.parse_args()

    os.makedirs('output/error_analysis', exist_ok=True)

    with open(args.log) as f:
        data = json.load(f)

    data_with_error = []
    for item in data:
        if item['recall']['5'] < 1.0:
            data_with_error.append(item)
    random.seed(1)
    sampled = random.sample(data_with_error, 100)

    with open('output/error_analysis/log.json', 'w') as f:
        json.dump(sampled, f)

    metrics = defaultdict(int)

    for item in sampled:
        nodes_in_gold_doc = json.loads(item['nodes_in_gold_doc'])
        question_decomposition = item['question_decomposition']
        print(f"[Query] {item['question']}")
        print(f"[Answer] {item['answer']}")
        recall_5 = item['recall']['5']
        print(f"[Recall@5] {recall_5}")

        print('[Supporting]')
        for p in item['supporting_docs']:
            print(p['title'])
        print()
        print('[Retrieved]')
        for r in item['retrieved'][:5]:
            print(r.split('\n')[0])
        print()

        facts_before_filter = item['rerank']['facts_before_rerank']
        facts_after_filter = item['rerank']['facts_after_rerank']
        phrases_before_filter = set([t[0] for t in facts_before_filter] + [t[2] for t in facts_before_filter])
        phrases_after_filter = set([t[0] for t in facts_after_filter] + [t[2] for t in facts_after_filter])

        phrases_before_filter_in_gold_doc = set()
        for f in facts_before_filter:
            for n in nodes_in_gold_doc:
                if f[0].lower() in n:
                    phrases_before_filter_in_gold_doc.add(f[0])
                if f[2].lower() in n:
                    phrases_before_filter_in_gold_doc.add(f[2])
        print(f"[LINKING] fact before filter: {facts_before_filter}")
        ratio_before = len(phrases_before_filter_in_gold_doc) / len(phrases_before_filter) if len(phrases_before_filter) > 0 else 0
        print('[LINKING] ratio of fact before filter in gold doc', ratio_before)
        print('[LINKING] phrases_before_filter_in_gold_doc', phrases_before_filter_in_gold_doc)

        phrases_after_filter_in_gold_doc = set()
        for f in facts_after_filter:
            for n in nodes_in_gold_doc:
                if f[0].lower() in n:
                    phrases_after_filter_in_gold_doc.add(f[0])
                if f[2].lower() in n:
                    phrases_after_filter_in_gold_doc.add(f[2])
        print(f"[LINKING] fact after filter: {facts_after_filter}")
        ratio_after = len(phrases_after_filter_in_gold_doc) / len(phrases_after_filter) if len(phrases_after_filter) > 0 else 0
        print('[LINKING] ratio of fact after filter in gold doc', ratio_after)
        print('[LINKING] phrases_after_filter_in_gold_doc', phrases_after_filter_in_gold_doc)

        num_hit_doc = 0
        for n in nodes_in_gold_doc:
            for f in facts_after_filter:
                if f[0].lower() in n or f[2].lower in n:
                    num_hit_doc += 1
                    break
        print(f"[Hop] {len(item['question_decomposition'])}")
        print('[LINKING] num doc hit by facts after filter', num_hit_doc)
        if num_hit_doc < len(item['question_decomposition']):
            metrics['[LINKING] not_hit_all_doc'] += 1

        no_doc_hit_by_neighbors = False
        if '1-hop_graph_for_linked_nodes' in item:
            one_hop_graph_for_linked_nodes = json.loads(item['1-hop_graph_for_linked_nodes'])
            neighbors_in_gold_doc = set()
            num_hit_doc_by_neighbors = 0
            for n in nodes_in_gold_doc:
                for triples in one_hop_graph_for_linked_nodes:
                    if triples[0].lower() in n:
                        num_hit_doc_by_neighbors += 1
                        break
            if num_hit_doc_by_neighbors < len(nodes_in_gold_doc):
                no_doc_hit_by_neighbors = True
                metrics['no_doc_hit_by_neighbors'] += 1
                print('[Graph] no doc hit by neighbors')

        if ratio_after < ratio_before and len(phrases_after_filter_in_gold_doc) > 0:
            print('[LINKING] Filter ratio decrease')
            metrics['filter_ratio_decrease'] += 1
        metrics[f'hop_{len(question_decomposition)}'] += 1
        metrics['ratio_0_before_filter'] += 1 if ratio_before == 0 else 0
        metrics['ratio_0_after_filter'] += 1 if ratio_after == 0 else 0

        if len(facts_after_filter) == 0:
            metrics['no_fact_after_filter'] += 1

        if ratio_after == 1.0 and no_doc_hit_by_neighbors is False:
            metrics['[PPR] potential issue'] += 1
            print('[PPR] potential issue')
        print()
        print()
    print(metrics)
