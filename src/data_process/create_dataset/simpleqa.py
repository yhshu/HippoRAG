import pandas as pd
import json

if __name__ == '__main__':
    csv_file_path = 'data/simple_qa_test_set.csv'

    # read csv and convert into json
    df = pd.read_csv(csv_file_path)
    df_json = df.to_json()

    data = []
    for i in range(len(df)):
        row = df.iloc[i]
        try:
            data.append({
                'question': row['problem'],
                'answer': row['answer'],
                'metadata': row['metadata']
            })
        except Exception as e:
            print(f'Error: {e}')
            print(row['metadata'])

    with open('data/simpleqa.json', 'w') as f:
        json.dump(data, f)
        print(f'Saving {len(data)} samples to simpleqa.json')