import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.initializers import Constant
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from model import X_test, y_test
from keras.models import load_model
from model import tokenizer
import keras.backend as K

model = load_model('Model/model_adam.h5')

# Select a specific test sample (in this example, the first sample)
sample_index = 0
dummy_input = X_test[sample_index].reshape(
    (1, -1))  # Reshape to match input shape

# Convert the labels to one-hot encoding
y_test_one_hot = to_categorical(y_test, num_classes=3)

# Get the one-hot encoded label for the sample
ground_truth_labels = y_test_one_hot[sample_index].reshape((1, 3))

# Define a range of values for each weight parameter
weight_range = np.linspace(-1, 1, 200)

# Create a placeholder model with the same architecture as your trained model
# Set the weights of the layers using the weights from your trained model
# For example, assuming 'trained_model' is your trained model:
placeholder_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100,
              input_length=250, mask_zero=True, embeddings_initializer=Constant(value=0.01)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(3, activation='softmax')
])
placeholder_model.set_weights(model.get_weights())

learning_rate = 0.00001

adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9,
                      beta_2=0.999, epsilon=1e-07, amsgrad=False)

placeholder_model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy',
                          metrics=['accuracy'])

# Initialize a 2D array to store loss values
loss_values = np.inf * np.ones((len(weight_range), len(weight_range)))

# Get the original weights of the dense layer
original_dense_weights = placeholder_model.layers[3].get_weights()

for i, weight1 in enumerate(weight_range):
    for j, weight2 in enumerate(weight_range):
        # Create a copy of the original weights
        new_dense_weights = original_dense_weights.copy()

        # Modify the weights for the dense layer
        new_dense_weights[0] = np.random.uniform(
            low=-abs(weight1), high=abs(weight1), size=(64, 3))
        new_dense_weights[1] = np.random.uniform(
            low=-abs(weight2), high=abs(weight2), size=(3,))

        # Set the modified weights
        placeholder_model.layers[3].set_weights(new_dense_weights)

        # Dummy input (shape should match the input shape of your model)
        dummy_input = np.zeros((1, 250))

        # Compute the loss
        loss = placeholder_model.evaluate(
            dummy_input, ground_truth_labels, verbose=0)[0]

        # Store the loss value
        loss_values[i, j] = loss

# Create a grid of (weight1, weight2) pairs
weight1_values, weight2_values = np.meshgrid(weight_range, weight_range)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the loss values
ax.plot_surface(weight1_values, weight2_values, loss_values)

# Set labels
ax.set_xlabel('Weight 1')
ax.set_ylabel('Weight 2')
ax.set_zlabel('Loss')

# Show the plot
plt.show()
