from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv(
    'Datasets/TRdata.csv', encoding='utf-16').dropna(subset=['Metinler', 'Duygular'])

X = df['Metinler'].values
y = df['Duygular'].values

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
# Set a fixed sequence length
X = pad_sequences(X, maxlen=250, truncating='pre', padding='pre', value=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

num_classes = 3  # Number of classes

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# np.savetxt('X_test.csv', X_test, delimiter=',')

# Define your learning rate for SGD
learning_rate = 0.01

# Specify the optimizer and its hyperparameters
adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9,
                      beta_2=0.999, epsilon=1e-07, amsgrad=False)

# Create the SGD optimizer without momentum
sgd_optimizer = SGD(learning_rate=learning_rate)

sdgm_optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

# Build the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=30,
              input_length=250, mask_zero=True, embeddings_initializer=Constant(value=0.5)),  # Embedding katmanına sabit başlangıç değeri
    # Add a 1D Convolutional Layer
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),  # Pooling layer to reduce dimensions
    Dense(3, activation='softmax')  # Dense katmanına sabit başlangıç değeri
])

if __name__ == '__main__':
    # Compile the model
    model.compile(optimizer=sdgm_optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train_one_hot, epochs=5, batch_size=32,
              validation_data=(X_test, y_test_one_hot), callbacks=[early_stopping])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
    print(f'Test accuracy: {test_accuracy}')

    # After training is complete, save the model
    model.save('Model/model_adam.h5')
