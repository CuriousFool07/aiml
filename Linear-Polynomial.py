
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)
pred_linear = linear.predict(X_test)

# 5. Polynomial Regression (degree 2)
poly = Pipeline([
    ("features", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression())
])
poly.fit(X_train, y_train)
pred_poly = poly.predict(X_test)

# 6. Compare models
results = pd.DataFrame({
    "Model": ["Linear", "Polynomial"],
    "MSE": [
        mean_squared_error(y_test, pred_linear),
        mean_squared_error(y_test, pred_poly)
    ],
    "MAE": [
        mean_absolute_error(y_test, pred_linear),
        mean_absolute_error(y_test, pred_poly)
    ],
    "R2 Score": [
        r2_score(y_test, pred_linear),
        r2_score(y_test, pred_poly)
    ]
})

print(results.round(4))

# 7. Plot actual vs predicted values
plt.scatter(y_test, pred_linear, label="Linear", alpha=0.5)
plt.scatter(y_test, pred_poly, label="Polynomial", alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--", label="Ideal prediction"
)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
