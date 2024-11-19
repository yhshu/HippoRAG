import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    from stark_qa import load_skb

    dataset_name = 'amazon'  # 'mag' / 'prime'
    skb = load_skb(dataset_name, download_processed=True)

    node_types = skb.node_type_lst()  # list node types in SKBs
    relation_types = skb.rel_type_lst()  # list relation types in SKBs

    print(skb.num_nodes(), skb.num_edges())  # count ndoes and edges in SKBs

    print(skb.get_node_ids_by_type(relation_types[0]))  # list node ids for a specific type
    print(skb.get_node_type_by_id(0))  # get node type by id

    print(skb.get_doc_info(0))  # get document info for node
    print(skb.get_neighbor_nodes(0, rel_types=relation_types[0]))  # list neighbors of a node by relation type
