from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/triangle_dataset.csv")
X = df[['a', 'b']]
y = df['c']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a nonlinear model (MLPRegressor)
model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/mlp_model.pkl")
print("MLP model saved to models/mlp_model.pkl")
