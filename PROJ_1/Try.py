import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "/Users/rohith/Downloads/diabetes.tab.txt"
df = pd.read_csv(file_path, delimiter="\t")  # Load tab-separated file

# Extract features and target
X = df.iloc[:, :-1].values  # Features (first 10 columns)
y = df.iloc[:, -1].values   # Target (last column: diabetes progression)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize parameters for gradient descent
m, n = X.shape
W = np.random.randn(n)  # Weights
b = np.random.randn()   # Bias
alpha = 0.01  # Learning rate
epochs = 1000

# Function to compute Mean Squared Error
def compute_mse(X, y, W, b):
    y_pred = np.dot(X, W) + b
    mse = mean_squared_error(y, y_pred)
    return mse

# Compute initial MSE before training
mse_before = compute_mse(X, y, W, b)

# Gradient Descent Function
def gradient_descent(X, y, W, b, alpha, epochs):
    m = len(y)
    for _ in range(epochs):
        y_pred = np.dot(X, W) + b
        error = y_pred - y

        # Compute gradients
        dW = (2/m) * np.dot(X.T, error)
        db = (2/m) * np.sum(error)

        # Update parameters
        W -= alpha * dW
        b -= alpha * db

    return W, b

# Model prediction function
def predict(user_input, W, b):
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)  # Scale input
    prediction = np.dot(user_input, W) + b  # Compute prediction
    return round(prediction[0], 2)

# Hardcoded input values
user_data = [41,1,21.6,87,183,103.2,70,3,3.8918,69]  

# Predict before training
prediction_before = predict(user_data, W, b)

# Train model using gradient descent
W, b = gradient_descent(X, y, W, b, alpha, epochs)

# Compute MSE after training
mse_after = compute_mse(X, y, W, b)

# Predict after training
prediction_after = predict(user_data, W, b)

# Print results
print(f"📊 MSE Before Training: {mse_before:.4f}")
print(f"📊 MSE After Training: {mse_after:.4f}")
print(f"📉 Change in MSE: {mse_before - mse_after:.4f}\n")

print(f"🔵 Prediction Before Training: {prediction_before}")
print(f"🩺 Prediction After Training: {prediction_after}")

# Set a threshold for diabetes risk
if prediction_after > 140:  # Adjust threshold based on dataset analysis
    print("⚠️ Patient may have diabetes")
else:
    print("✅ Patient is at low risk of diabetes")
