import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    from stark_qa import load_skb

    dataset_name = 'amazon'  # 'amazon' / 'mag' / 'prime'
    skb = load_skb(dataset_name, download_processed=True)

    node_types = skb.node_type_lst()  # list node types in SKBs
    relation_types = skb.rel_type_lst()  # list relation types in SKBs

    print(skb.num_nodes(), skb.num_edges())  # count ndoes and edges in SKBs

    print(skb.get_node_ids_by_type(node_types[0]))  # list node ids for a specific type
    print(skb.get_node_type_by_id(0))  # get node type by id

    print(skb.get_edge_ids_by_type(relation_types[0]))
    print(skb.get_edge_type_by_id(0))

    print(skb.get_doc_info(0))  # get document info for node
    print(skb.get_neighbor_nodes(0, edge_type=relation_types[0]))  # list neighbors of a node by relation type

    print(skb.node_info[0])

    from stark_qa import load_qa

    dataset_name = 'amazon'  # 'amazon' / 'mag' / 'prime'
    ROOT = None
    qa_dataset = load_qa(dataset_name, root=ROOT, human_generated_eval=False)
    human_eval_set = load_qa(dataset_name, root=ROOT, human_generated_eval=True)
    idx_split = qa_dataset.get_idx_split()
    print(qa_dataset[1])  # query, query_id, answer_ids, metadata

    print('#train', len(idx_split['train']))
    print('#dev', len(idx_split['dev']))
    print('#test', len(idx_split['test']))
