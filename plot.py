from sklearn.manifold import TSNE
from model import model, X_train, y_train
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate synthetic input data
x_synthetic = 1  # YAPILACAK

# Use the trained neural network to make predictions
y_predicted = model.predict(x_synthetic)

# Plot input vs. predicted output
plt.scatter(x_synthetic, y_predicted, label='Predicted Output')
plt.xlabel('Input')
plt.ylabel('Predicted Output')
plt.legend()
plt.show()
