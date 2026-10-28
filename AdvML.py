
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create models
svm = make_pipeline(StandardScaler(), SVC())
rf = RandomForestClassifier(n_estimators=100, random_state=42)
ada = AdaBoostClassifier(n_estimators=50, random_state=42)

models = {
    "SVM": svm,
    "Random Forest": rf,
    "AdaBoost": ada
}

results = []

# 4. Train and evaluate models
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

print("Model comparison:")
print(pd.DataFrame(results).round(4))

# 5. Cross-validation on SVM
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    svm, X_train, y_train, cv=cv, scoring="f1_weighted"
)
print("CV F1 scores:", cv_scores)
print("Mean CV F1:", cv_scores.mean())

# 6. Hyperparameter tuning using GridSearchCV
param_grid = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    param_grid,
    cv=cv,
    scoring="f1_weighted"
)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV F1 score:", grid.best_score_)

# 7. Evaluate tuned SVM on test data
best_model = grid.best_estimator_
pred = best_model.predict(X_test)

print("Tuned SVM test accuracy:", accuracy_score(y_test, pred))
print("Tuned SVM test F1:", f1_score(y_test, pred, average="weighted"))

# 8. Plot model comparison
comparison = pd.DataFrame(results).set_index("Model")
comparison[["Accuracy", "F1 Score"]].plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
