import gradio as gr
import joblib
import numpy as np

# Load the trained MLP model
model = joblib.load("models/mlp_model.pkl")

# Define the prediction function
def predict_hypotenuse(a, b):
    a = float(a)
    b = float(b)
    prediction = model.predict([[a, b]])[0]
    return round(prediction, 4)

# Gradio interface
demo = gr.Interface(
    fn=predict_hypotenuse,
    inputs=[
        gr.Number(label="Side a"),
        gr.Number(label="Side b")
    ],
    outputs=gr.Number(label="Predicted Hypotenuse (c)"),
    title="Pythagoras Predictor",
    description="Enter the two perpendicular sides of a right triangle to predict the hypotenuse using a trained MLP model."
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
