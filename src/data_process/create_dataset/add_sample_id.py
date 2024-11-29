import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    file_path = f'data/{args.dataset}.json'
    data = json.load(open(file_path, 'r'))
    for i, d in enumerate(data):
        if 'id' in d or '_id' in d:
            continue
        d['id'] = i
    json.dump(data, open(file_path, 'w'), indent=2)
