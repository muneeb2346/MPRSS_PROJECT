import pandas as pd


class AndroidMalwareDataLoader:

    def __init__(self):
        self.datasets    = {}
        self.dataset_info = {}

    def load_csv_dataset(self, filepath, label_column='label', **kwargs):
        try:
            df = pd.read_csv(filepath, **kwargs)
            print(f"Loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")

            if label_column not in df.columns:
                possible_labels = [
                'labeled', 'label', 'class', 'malware', 'type', 'Category',
                'is_malicious', 'Label', 'Class'
                ]
                for col in possible_labels:
                    if col in df.columns:
                        label_column = col
                        break
                else:
                    # If no label column found, assume last column is label
                    label_column = df.columns[-1]

            print(f"Label column detected: '{label_column}'")
            return df, label_column

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None


if __name__ == "__main__":
    loader = AndroidMalwareDataLoader()
    df, label_col = loader.load_csv_dataset('datasets/Android_Malware.csv')
    if df is not None:
        print(f"\nDataset preview:\n{df.head()}")
        print(f"\nLabel distribution:\n{df[label_col].value_counts()}")