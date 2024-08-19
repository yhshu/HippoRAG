import argparse
import json
import os
import pickle

import rdflib
from tqdm import tqdm

from src.hipporag import HippoRAG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sample')
    parser.add_argument('--extractor', type=str)
    parser.add_argument('--retriever', type=str)
    args = parser.parse_args()

    # read triples
    hipporag = HippoRAG(args.dataset, extractor_name=args.extractor, graph_creating_retriever_name=args.retriever)
    triples = pickle.load(
        open('output/{}_{}_graph_relation_dict_{}_{}_{}.{}.subset.p'.format(args.dataset, hipporag.graph_type,
                                                                            hipporag.phrase_type, hipporag.extraction_type,
                                                                            hipporag.graph_creating_retriever_name_processed,
                                                                            hipporag.version), 'rb'))  # (h, t): tuple -> r: str

    # create graph
    nodes = set()
    ns = rdflib.Namespace(f"https://github.com/OSU-NLP-Group/HippoRAG/{args.dataset}/")
    for key, value in triples.items():
        nodes.add(key[0])
        nodes.add(key[1])

    nodes = list(nodes)
    node_to_mid = {node: ns[f"Node{i}"] for i, node in enumerate(nodes)}
    edge_to_mid = {edge: ns[f"Edge{i}"] for i, edge in enumerate(triples.values())}
    mid_to_node = {v: k for k, v in node_to_mid.items()}
    rdf_graph = rdflib.Graph()
    for node in nodes:
        if not node.isdigit():
            rdf_graph.add((ns[node_to_mid[node]], rdflib.URIRef('http://www.w3.org/2000/01/rdf-schema#label'), rdflib.Literal(node)))
    for key, value in tqdm(triples.items(), desc='Adding triples to RDF graph', total=len(triples)):
        rdf_graph.add((ns[node_to_mid[key[0]]], ns['_'.join(value.split(' '))], ns[node_to_mid[key[1]]]))

    rdf_graph.bind(f"{args.dataset}", ns)
    rdf_graph.bind("rdfs", rdflib.RDFS)

    os.makedirs('output/ttl_kg', exist_ok=True)
    ttl_output_path = 'output/ttl_kg/{}_{}_graph_{}_{}_{}.{}.ttl'.format(args.dataset, hipporag.graph_type, hipporag.phrase_type, hipporag.extraction_type,
                                                                         hipporag.graph_creating_retriever_name_processed, hipporag.version)
    print('Saving TTL file to', ttl_output_path)
    rdf_graph.serialize(destination=ttl_output_path, format='turtle')

    # save node_to_mid
    node_to_mid_path = 'output/ttl_kg/{}_{}_graph_{}_{}_{}.{}.node_to_mid.json'.format(args.dataset, hipporag.graph_type, hipporag.phrase_type, hipporag.extraction_type,
                                                                                       hipporag.graph_creating_retriever_name_processed, hipporag.version)
    print('Saving node_to_mid to', node_to_mid_path)
    json.dump(node_to_mid, open(node_to_mid_path, 'w'))

    # save mid_to_node
    mid_to_node_path = 'output/ttl_kg/{}_{}_graph_{}_{}_{}.{}.mid_to_node.json'.format(args.dataset, hipporag.graph_type, hipporag.phrase_type, hipporag.extraction_type,
                                                                                       hipporag.graph_creating_retriever_name_processed, hipporag.version)
    print('Saving mid_to_node to', mid_to_node_path)
    json.dump(mid_to_node, open(mid_to_node_path, 'w'))

    print('Done')
