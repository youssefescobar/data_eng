import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

def feature_selection(input, output):
    df = pd.read_csv(input)

    df = df.sample(frac=0.5, random_state=42)
    print("Dropped 50% of the data.")

    new_shape = df.shape
    print(f"New shape: {new_shape}")
    with open("./output/feature_eng/feature_selection_info.txt", "w") as f:
        f.write(f"New shape after dropping 50% of the data: {new_shape}\n")

    X = df.drop(['Class', 'id'], axis=1, errors='ignore')
    y = df['Class']

    model = RandomForestClassifier()
    model.fit(X, y)

    feature_importances = model.feature_importances_
    plt.barh(X.columns, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance from Random Forest')
    plt.tight_layout()
    plt.savefig("./output/feature_eng/random_forest_features.png")
    plt.close()

    selected_fet = ['V14', 'V12', 'V10', 'V4', 'V17']
    X_selected = X[selected_fet]

    print("Final selected features:", selected_fet)
    with open("./output/feature_eng/final_selected_features.txt", "w") as f:
        f.write("Final selected features:\n")
        f.write("\n".join(selected_fet))

    X_selected.to_csv(output, index=False)
    print("Selected features saved to", output)

if __name__ == "__main__":
    feature_selection("preprocessed.csv", "selected_features.csv")
    print("data ingestion, preprocessing, feature engineering is done.")
