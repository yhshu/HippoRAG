import argparse
import json

from src.openie_with_retrieval_option_parallel import load_corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_passages', type=str)
    args = parser.parse_args()

    run_ner = True
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(args.dataset, args.model_name, args.num_passages, run_ner)
    # todo
    extra_info_json = None
    # extra_info_json = {"docs": new_json,
    #                        "ents_by_doc": ents_by_doc,
    #                        "avg_ent_chars": avg_ent_chars,
    #                        "avg_ent_words": avg_ent_words,
    #                        "num_tokens": lm_total_tokens,
    #                        "approx_total_tokens": approx_total_tokens,
    #                        }
    output_path = 'output/openie{}_results_{}.json'.format(dataset_name, arg_str)
    json.dump(extra_info_json, open(output_path, 'w'))
    print('OpenIE saved to', output_path)