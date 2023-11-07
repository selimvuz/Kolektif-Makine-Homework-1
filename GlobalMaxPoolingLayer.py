from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from keras.models import load_model
from model import X, y
from keras.models import Model
import matplotlib.pyplot as plt

model = load_model('Model/model_v22.h5')

intermediate_layer_model = Model(
    inputs=model.input, outputs=model.layers[2].output)

# Assuming 'X' is your input data
intermediate_output = intermediate_layer_model.predict(X)

# Flatten the output
flattened_output = intermediate_output.reshape(
    intermediate_output.shape[0], -1)

# Perform PCA
# You can choose the number of components you want to keep
pca = PCA(n_components=2)
pca_result = pca.fit_transform(flattened_output)

# Plot PCA results
# Assuming 'y' is the labels for your data
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Text Features')
plt.colorbar()
plt.show()
