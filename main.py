import pandas as pd

if __name__ == "__main__":
    melb_file_path = './datasets/melb_data.csv'
    melb_data = pd.read_csv(melb_file_path)
    print(melb_data.describe())