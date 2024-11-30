import json

from src.processing import query_data_has_duplication, corpus_has_duplication

if __name__ == '__main__':
    datasets = [
        'musique',
        '2wikimultihopqa',
        'hotpotqa',
        "beir_scifact_test",
        "beir_scidocs_test",
        "beir_nfcorpus_dev_324",
        "beir_fiqa_dev_500",
        "beir_climate-fever_test_500",
        "beir_nq_test_500",
        "beir_trec-covid_test_50",
        "beir_dbpedia-entity_dev_67",
        "beir_fever_dev_500",
        "musique_train_1000",
        "musique_dev_1000",
        "2wikimultihopqa_train_1000",
        "2wikimultihopqa_dev_1000",
        "beir_msmarco_train_1000",
        "beir_msmarco_dev_1000",
        "hippoqa_biomed_ans_dev",
        "hippoqa_books_ans_dev",
        "hippoqa_movie_ans_dev",
        "lveval",
        "narrativeqa_dev_10_doc"
    ]
    for dataset in datasets:
        data = json.load(open(f'data/{dataset}.json'))
        duplication, d = query_data_has_duplication(data, True, False)
        if duplication:
            print(f"dataset {dataset} has query duplication, unique: {len(d)}, all: {len(data)}")

        corpus = json.load(open(f"data/{dataset}_corpus.json"))
        duplication = corpus_has_duplication(corpus)
        if duplication:
            print(f"dataset {dataset} has passage duplication")
