import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("./data/creditcard_2023.csv")
df = df.sample(frac=0.5, random_state=42)
X = df.drop(['Class', 'id'], axis=1, errors='ignore')
y = df['Class']

selected_fet = ['V14', 'V12', 'V10', 'V4', 'V17']
X_selected = X[selected_fet]  

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, 
                           cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


y_pred = best_knn.predict(X_test)

with open("./output/model/model_performance.txt", "w") as f:
    f.write("Best Parameters:\n")
    f.write(str(grid_search.best_params_) + "\n\n")
    
    f.write("Accuracy: {:.4f}\n".format(accuracy_score(y_test, y_pred)))
    
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))
    
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)) + "\n")
    
    if len(set(y)) == 2:  
        y_pred_proba = best_knn.predict_proba(X_test)[:, 1]
        f.write("\nROC-AUC Score: {:.4f}\n".format(roc_auc_score(y_test, y_pred_proba)))

X_selected_20000 = X_selected.iloc[:20000, :]
y_20000 = y.iloc[:20000]  
train_sizes, train_scores, val_scores = learning_curve(best_knn, X_selected_20000, y_20000, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue')
plt.plot(train_sizes, val_mean, label='Validation Accuracy', color='orange')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.2)
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./output/model/learning_curve.png")
plt.close()

print("Model evaluation completed and results saved.")
