
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# 1. Load dataset
data = load_iris(as_frame=True)
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB()
}

results = []

# 4. Train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted"),
        "Recall": recall_score(y_test, pred, average="weighted"),
        "F1 Score": f1_score(y_test, pred, average="weighted")
    })

    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=data.target_names
    ).plot()
    plt.title(name)
    plt.show()

# 5. Plot Decision Tree
tree_model = models["Decision Tree"]
plt.figure(figsize=(12, 7))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=data.target_names,
    filled=True
)
plt.title("Decision Tree")
plt.show()

# 6. Compare results
print(pd.DataFrame(results).round(4))
