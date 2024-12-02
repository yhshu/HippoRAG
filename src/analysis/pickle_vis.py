import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        data = pickle.load(f)
    print(type(data))
    print(len(data))