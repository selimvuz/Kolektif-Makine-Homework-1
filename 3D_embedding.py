# Initialize t-SNE with desired parameters
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from keras.models import load_model
from model import X_train, y_train
from keras.models import Model
import matplotlib.pyplot as plt

model = load_model('Model/model_v8.h5')

# Create a new model that outputs the embeddings from the Dense layer
embedding_model = Model(
    inputs=model.input, outputs=model.get_layer('embedding').output)

# Use the new model to predict on your input data
embeddings = embedding_model.predict(X_train)

# Flatten the output
num_samples, sequence_length, embedding_dim = embeddings.shape
flattened_output = embeddings.reshape(
    (num_samples, sequence_length * embedding_dim))

# Apply PCA to reduce the dimensions of the embeddings
pca = PCA(n_components=2)
pca_output = pca.fit_transform(flattened_output)

# Initialize t-SNE with desired parameters
tsne = TSNE(n_components=2, random_state=42)

# Apply t-SNE to your data
embeddings_2d = tsne.fit_transform(pca_output)

labels = y_train

# Visualize the embeddings
plt.scatter(embeddings_2d[:, 0],
            embeddings_2d[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.show()
