from datasets import load_dataset

if __name__ == '__main__':
    dataset = load_dataset('ms_marco', 'v2.1')

    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']
