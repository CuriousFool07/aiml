
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# 1. Load Digits dataset
data = load_digits()
X = data.data
y = data.target

print("Dataset shape:", X.shape)

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Model using all 64 features
model_full = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)
model_full.fit(X_train, y_train)
pred_full = model_full.predict(X_test)

# 4. Model using only 2 PCA components
model_pca = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(max_iter=5000)
)
model_pca.fit(X_train, y_train)
pred_pca = model_pca.predict(X_test)

# 5. Compare evaluation metrics
def get_metrics(y_true, pred, name):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(
            y_true, pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true, pred, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, pred, average="weighted", zero_division=0
        )
    }

results = pd.DataFrame([
    get_metrics(y_test, pred_full, "All 64 features"),
    get_metrics(y_test, pred_pca, "PCA: 2 components")
])

print(results.round(4))

# 6. Plot the two principal components
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Dataset after PCA")
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", pca.explained_variance_ratio_.sum())
