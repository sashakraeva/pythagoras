import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_csv("data/triangle_dataset.csv")
X = df[['a', 'b']]
y = df['c']

model = joblib.load("models/linear_regression_model.pkl")
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("True Hypotenuse (c)")
plt.ylabel("Predicted Hypotenuse (c)")
plt.title("Model Prediction vs True Values")
plt.grid(True)
plt.show()
plt.savefig("results/pred_vs_true_01.png", dpi=300)

