import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("data/triangle_dataset.csv")
X = df[['a', 'b']]
y = df['c']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/linear_regression_model.pkl")
print("Model saved to models/linear_regression_model.pkl")
