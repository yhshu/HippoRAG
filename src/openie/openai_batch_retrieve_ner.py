import argparse

from openai import OpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_id', type=str, help="OpenAI file ID to retrieve", required=True)
    args = parser.parse_args()

    client = OpenAI()
    content = client.files.content(file_id=args.file_id)

    # save file to do OpenIE after NER
