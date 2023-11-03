from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

loss_values = [-11.0380, -41.2647, -75.9776, -111.8173, -
               155.9470, -192.5085, -235.2972, -271.7552, -309.7030, -350.2706]

# Create corresponding epochs
epochs = list(range(1, len(loss_values) + 1))

# Combine epochs and loss values
combined_data = np.column_stack((epochs, loss_values))

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=5)
tsne_output = tsne.fit_transform(combined_data)

# Visualize the t-SNE output
plt.scatter(tsne_output[:, 0], tsne_output[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Loss Values t-SNE Visualization')
plt.show()
