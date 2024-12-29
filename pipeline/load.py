import pandas as pd
import os
import io

def load_dataset(file):
    df = pd.read_csv(file)
    print("Data loaded successfully.")
    print(df.head())
    with open("./output/info/head.txt", "w") as f:
        f.write(df.head().to_string())

    print("Dataset info:")
    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    info_text = info_buffer.getvalue()
    print(info_text)
    with open("./output/info/info.txt", "w") as f:
        f.write(info_text)
    

    print("Dataset description:")
    description = df.describe().to_string()
    print(description)
    with open("./output/info/description.txt", "w") as f:
        f.write(description)

    df.to_csv("loaded.csv", index=False)
    print("Data saved to loaded.csv")

if __name__ == "__main__":
    file_path = './data/creditcard_2023.csv'
    load_dataset(file_path)
    os.system("python preprocessing.py")