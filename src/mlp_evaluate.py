import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset and model
df = pd.read_csv("data/triangle_dataset.csv")
X = df[['a', 'b']]
y = df['c']

model = joblib.load("models/mlp_model.pkl")
y_pred = model.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # ideal line
plt.xlabel("True Hypotenuse (c)")
plt.ylabel("Predicted Hypotenuse (c)")
plt.title("MLP: Predicted vs True c")
plt.grid(True)
plt.savefig("results/pred_vs_true_mlp.png", dpi=300)
plt.show()
