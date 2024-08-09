import json
from collections import defaultdict


def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_metrics(intermediate, selected_nodes):
    precision = len(set(intermediate) & set(selected_nodes)) / len(set(selected_nodes))
    recall = len(set(intermediate) & set(selected_nodes)) / len(set(intermediate))
    any_recall = 1.0 if len(set(intermediate) & set(selected_nodes)) > 0 else 0
    all_recall = 1.0 if len(set(intermediate) - set(selected_nodes)) == 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, any_recall, all_recall, f1_score


def process_data(data1, data2):
    num_sample = 0
    metrics = defaultdict(float)

    for idx in range(len(data1)):
        llm_selected_nodes1 = data1[idx]["llm_selected_nodes"]
        llm_selected_nodes2 = data2[idx]["llm_selected_nodes"]

        num_sample += 1

        print("[question]", data1[idx]["question"])
        intermediate_answers = [item["answer"] for item in data1[idx]["question_decomposition"]]
        print("[intermediate]", intermediate_answers)
        processed_intermediate = [x.lower().strip('.') for x in intermediate_answers]
        print("[answer]", data1[idx]["answer"])

        print("nodes1", llm_selected_nodes1)
        print("nodes2", llm_selected_nodes2)
        num_node1 = len(llm_selected_nodes1)
        num_node2 = len(llm_selected_nodes2)

        # Calculate metrics for both sets of selected nodes
        precision1, recall1, any_recall1, all_recall1, f1_score1 = calculate_metrics(processed_intermediate, llm_selected_nodes1)
        precision2, recall2, any_recall2, all_recall2, f1_score2 = calculate_metrics(processed_intermediate, llm_selected_nodes2)

        # Calculate upperbound for extracted nodes in the gold doc
        all_nodes_in_doc1 = set([node for node_list in json.loads(data1[idx]["nodes_in_retrieved_doc"]) for node in node_list])
        all_nodes_in_doc2 = set([node for node_list in json.loads(data2[idx]["nodes_in_retrieved_doc"]) for node in node_list])

        recall_ub1 = len(set(processed_intermediate).intersection(all_nodes_in_doc1)) / len(set(processed_intermediate))
        recall_ub2 = len(set(processed_intermediate).intersection(all_nodes_in_doc2)) / len(set(processed_intermediate))

        any_recall_ub1 = 1.0 if len(set(processed_intermediate).intersection(all_nodes_in_doc1)) > 0 else 0
        any_recall_ub2 = 1.0 if len(set(processed_intermediate).intersection(all_nodes_in_doc2)) > 0 else 0

        # Update metrics
        for key, value in zip(['num_node1', 'precision1', 'recall1', 'any_recall1', 'all_recall1', 'f1_score1', 'recall_ub1', 'any_recall_ub1'],
                              [num_node1, precision1, recall1, any_recall1, all_recall1, f1_score1, recall_ub1, any_recall_ub1]):
            metrics[key] += value
        for key, value in zip(['num_node2', 'precision2', 'recall2', 'any_recall2', 'all_recall2', 'f1_score2', 'recall_ub2', 'any_recall_ub2'],
                              [num_node2, precision2, recall2, any_recall2, all_recall2, f1_score2, recall_ub2, any_recall_ub2]):
            metrics[key] += value

        print()

    return num_sample, metrics


if __name__ == '__main__':
    log1 = load_data(
        "output/ircot/ircot_results_musique_hipporag_R_GritLM_GritLM-7B_L_GritLM_GritLM-7B_demo_1_E_gpt-4o_no_ensemble_step_1_top_10_ppr_damping_0.1_sim_thresh_0.8_LT_5.json")
    log2 = load_data(
        "output/ircot/ircot_results_musique_hipporag_R_GritLM_GritLM-7B_L_GritLM_GritLM-7B_demo_1_E_gpt-4o-mini_no_ensemble_step_1_top_10_ppr_damping_0.1_sim_thresh_0.8_LT_5.json")

    num_sample, metrics = process_data(log1, log2)

    print("num_sample", num_sample)
    for key in metrics:
        print(key, round(metrics[key] / num_sample, 3))
