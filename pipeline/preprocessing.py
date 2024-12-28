# preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

def preprocess_data(input, output):
    df = pd.read_csv(input)

    null_counts = df.isnull().sum()
    with open("./output/preprocessing/null_counts.txt", "w") as f:
        f.write("Null value counts:\n")
        f.write(null_counts.to_string())

    duplicate_count = df.duplicated().sum()
    with open("./output/preprocessing/duplicate_count.txt", "w") as f:
        f.write(f"Duplicate rows count: {duplicate_count}\n")

    df = df.dropna()
    df = df.drop_duplicates()

    df.hist(figsize=(20, 15), bins=50)
    plt.tight_layout()
    plt.savefig("./output/preprocessing/histogram.png")
    plt.close()

    scaler = MinMaxScaler()
    if 'Amount' in df.columns:
        df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
        df.drop(columns=['Amount'], inplace=True)


    df.hist(figsize=(20, 15), bins=50)
    plt.tight_layout()
    plt.savefig("./output/preprocessing/transformed_histogram.png")
    plt.close()

    print("Preprocessing done.")
    print(df.head())
    with open("./output/preprocessing/preprocessed_head.txt", "w") as f:
        f.write(df.head().to_string())

    df.to_csv(output, index=False)
    print("Data saved to", output)

if __name__ == "__main__":
    preprocess_data("loaded.csv", "preprocessed.csv")
    os.system("python feature_selection.py")
