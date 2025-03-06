import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent macOS issues
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Flask app setup
app = Flask(__name__)

# Load dataset
file_path = "static/diabetes.tab.txt"
df = pd.read_csv(file_path, delimiter="\t")

# Extract features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize parameters for gradient descent
m, n = X.shape
W = np.random.randn(n)
b = np.random.randn()
alpha = 0.01
epochs = 1000

# Function to compute MSE
def compute_mse(X, y, W, b):
    y_pred = np.dot(X, W) + b
    return mean_squared_error(y, y_pred)

# Gradient Descent function
def gradient_descent(X, y, W, b, alpha, epochs):
    m = len(y)
    mse_history = []

    for _ in range(epochs):
        y_pred = np.dot(X, W) + b
        error = y_pred - y

        # Compute gradients
        dW = (2/m) * np.dot(X.T, error)
        db = (2/m) * np.sum(error)

        # Update weights
        W -= alpha * dW
        b -= alpha * db

        # Store MSE history
        mse_history.append(mean_squared_error(y, np.dot(X, W) + b))

    return W, b, mse_history

# Model prediction function
def predict(user_input, W, b):
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)
    prediction = np.dot(user_input, W) + b
    return round(prediction[0], 2)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    mse_plot = None

    if request.method == "POST":
        try:
            # Get user inputs
            user_data = [float(request.form[key]) for key in request.form.keys()]

            # Predict before training
            prediction_before = predict(user_data, W, b)

            # Train model
            trained_W, trained_b, mse_history = gradient_descent(X, y, W, b, alpha, epochs)

            # Compute MSE before and after training
            mse_before = compute_mse(X, y, W, b)
            mse_after = compute_mse(X, y, trained_W, trained_b)

            # Predict after training
            prediction_after = predict(user_data, trained_W, trained_b)

            # Risk level assessment
            risk_status = "⚠️ Patient may have diabetes" if prediction_after > 140 else "✅ Patient is at low risk"

            # Generate MSE plot
            plt.figure(figsize=(8, 5))
            plt.plot(range(epochs), mse_history, label="MSE Over Iterations", color="blue")
            plt.xlabel("Epochs")
            plt.ylabel("Mean Squared Error")
            plt.title("MSE Reduction with Gradient Descent")
            plt.legend()
            plt.grid(True)

            # Convert plot to base64 image
            img = io.BytesIO()
            plt.savefig(img, format="png")
            plt.close()  # Close the plot to prevent memory leak
            img.seek(0)
            mse_plot = base64.b64encode(img.getvalue()).decode()

            # Store results
            result = {
                "mse_before": round(mse_before, 4),
                "mse_after": round(mse_after, 4),
                "change_in_mse": round(mse_before - mse_after, 4),
                "prediction_before": prediction_before,
                "prediction_after": prediction_after,
                "risk_status": risk_status,
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result, mse_plot=mse_plot)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
