# Initialize t-SNE with desired parameters
from sklearn.manifold import TSNE
from model import model, X_train, y_train
from keras.models import Model
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)

# Create a new model that outputs the embeddings from the LSTM layer
embedding_model = Model(
    inputs=model.input, outputs=model.get_layer('lstm').output)

# Use the new model to predict on your input data
lstm_embeddings = embedding_model.predict(X_train)

# Apply t-SNE to your data
embeddings_2d = tsne.fit_transform(lstm_embeddings)

labels = y_train

# Visualize the embeddings
plt.scatter(embeddings_2d[:, 0],
            embeddings_2d[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.show()