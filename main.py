import numpy as np

# Veri seti ve model parametreleri
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 3.5, 4.5, 5])

# Model parametreleri
w = 0.5
b = 0.5

# Hiperparametreler
learning_rate = 0.01
momentum = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

# MSE hesaplama fonksiyonu


def compute_mse(X, y, w, b):
    y_pred = X*w + b
    mse = np.mean((y - y_pred)**2)
    return mse


# Stokastik Gradient Descent (SGD)
print("Stokastik Gradient Descent")

for epoch in range(100):
    for i in range(len(X)):
        y_pred = w*X[i] + b
        error = y_pred - y[i]
        w = w - learning_rate * error * X[i]
        b = b - learning_rate * error

    # Evaluate and print MSE
    mse = compute_mse(X, y, w, b)
    print(f"Epoch {epoch+1}, MSE: {mse}")

# Stokastik Gradient Descent with Momentum
w = 0.5
b = 0.5
v_w = 0
v_b = 0

print("Stokastik Gradient Descent with Momentum")

for epoch in range(100):
    for i in range(len(X)):
        y_pred = w*X[i] + b
        error = y_pred - y[i]
        v_w = momentum * v_w - learning_rate * error * X[i]
        v_b = momentum * v_b - learning_rate * error
        w = w + v_w
        b = b + v_b

    # Evaluate and print MSE
    mse = compute_mse(X, y, w, b)
    print(f"Epoch {epoch+1}, MSE: {mse}")

# Stokastik Adam
w = 0.5
b = 0.5
m_w = 0
m_b = 0
v_w = 0
v_b = 0

print("Stokastik Adam")

for epoch in range(100):
    for i in range(len(X)):
        y_pred = w*X[i] + b
        error = y_pred - y[i]
        m_w = beta1 * m_w + (1 - beta1) * error * X[i]
        m_b = beta1 * m_b + (1 - beta1) * error
        v_w = beta2 * v_w + (1 - beta2) * (error * X[i])**2
        v_b = beta2 * v_b + (1 - beta2) * error**2
        m_w_hat = m_w / (1 - beta1**(i+1))
        m_b_hat = m_b / (1 - beta1**(i+1))
        v_w_hat = v_w / (1 - beta2**(i+1))
        v_b_hat = v_b / (1 - beta2**(i+1))
        w = w - learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        b = b - learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    # Evaluate and print MSE
    mse = compute_mse(X, y, w, b)
    print(f"Epoch {epoch+1}, MSE: {mse}")
