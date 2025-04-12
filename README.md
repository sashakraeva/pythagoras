# 🧠 Predicting the Hypotenuse with Machine Learning

This project uses machine learning to predict the hypotenuse (c) of a right-angled triangle using sides a and b, based on Pythagoras' Theorem. It's part of the **ML for Robotic Fabrication** course.

---

## 🎯 Objective

Approximate the mathematical formula:
\[
c = \sqrt{a^2 + b^2}
\]
using supervised machine learning.

---

## 🗂️ Folder Structure

```
pythagoras/
├── data/
│   └── triangle_dataset.csv         # Generated dataset
├── models/
│   └── linear_regression_model.pkl  # Trained ML model
├── results/
│   └── pred_vs_true_01.png          # Visualization of predictions after 1st training
├── src/
│   ├── generate_dataset.py          # Create synthetic dataset for triangle_dataset.csv 
│   ├── train_model.py               # Train the ML model
│   └── evaluate_model.py            # Evaluate and visualize predictions
├── requirements.txt                 # Python dependencies
└── README.md                        
```

---

## ⚙️ Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 📊 Workflow

### 1. Generate Dataset

```bash
python src/generate_dataset.py
```

Creates `data/triangle_dataset.csv` with random side lengths and computed hypotenuse.

---

### 2. Train Model

```bash
python src/train_model.py
```

Trains a Linear Regression model and saves it to `models/linear_regression_model.pkl`.

---

### 3. Evaluate Model

```bash
python src/evaluate_model.py
```

Prints metrics and saves the prediction plot to `results/pred_vs_true.png`.

---

## 📈 Results

### 🔍 Evaluation Metrics
| Metric | Value  |
|--------|------------------|
| MSE    | 36.2317          |
| R²     | 0.9567          |

The model performs extremely well because it’s learning a deterministic mathematical relationship.

---

### 📷 Prediction Plot

This plot shows predicted vs true hypotenuse values:

![Prediction vs True](results/pred_vs_true_01.png)

---

## 🚀 Next Steps

- Add Gaussian noise to `a` and `b` to simulate sensor error.
- Train a neural network regressor (e.g., `MLPRegressor`).
- Deploy as a simple web tool using Streamlit or Gradio.

---

## ✍️ Author

Developed as part of an individual assignment for **ML for Robotic Fabrication**.
