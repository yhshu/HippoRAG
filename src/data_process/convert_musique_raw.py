import argparse
import json
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/raw/musique')
    args = parser.parse_args()

    for split in ['train', 'dev', 'test']:
        path = os.path.join(args.dir, f'musique_ans_v1.0_{split}.jsonl')
        # read musique raw jsonl file
        split_data = []
        with open(path, 'r') as f:
            raw = f.readlines()
            for line in raw:
                split_data.append(json.loads(line))

        output_path = f'data/musique_{split}_raw.json'
        with open(output_path, 'w') as f:
            json.dump(split_data, f)
        print(f'Saving {split} ({len(split_data)}) to {output_path}')
