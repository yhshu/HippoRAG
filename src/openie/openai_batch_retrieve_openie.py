import argparse
import json

from openai import OpenAI
from tqdm import tqdm

from src.openie_with_retrieval_option_parallel import load_corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--file_id', type=str, help="OpenAI file ID to retrieve", required=True)
    parser.add_argument('--num_passages', type=str, default='all')
    args = parser.parse_args()

    client = OpenAI(max_retries=5, timeout=60)
    content = client.files.content(file_id=args.file_id)

    # save file to do OpenIE after NER
    lines = content.text.strip().split("\n")
    print('content length:', len(lines))

    run_ner = True
    arg_str, dataset_name, flags_present, num_passages, retrieval_corpus = load_corpus(args.dataset, args.model_name, args.num_passages, run_ner)

    response = {} # custom id -> response, note that batch API doesn't guarantee order
    for line in tqdm(lines):
        item = json.loads(line)
        custom_id = item['custom_id']
        response[custom_id] = item['response']
        # todo
    # extra_info_json = {"docs": extraction_by_doc,
    #                    "ents_by_doc": ents_by_doc,
    #                    "avg_ent_chars": avg_ent_chars,
    #                    "avg_ent_words": avg_ent_words,
    #                    "num_tokens": lm_total_tokens,
    #                    "approx_total_tokens": approx_total_tokens,
    #                    }
    # output_path = 'output/openie{}_results_{}.json'.format(dataset_name, arg_str)
    # json.dump(extra_info_json, open(output_path, 'w'))
    # print('OpenIE saved to', output_path)