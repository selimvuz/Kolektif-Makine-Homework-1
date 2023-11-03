from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pandas as pd

# Load the data
df = pd.read_csv('Twitter_Data.csv').dropna(subset=['Metinler', 'Duygular'])

X = df['Metinler'].values
y = df['Duygular'].values

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=100)  # Set a fixed sequence length

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# np.savetxt('X_test.csv', X_test, delimiter=',')

# Specify the optimizer and its hyperparameters
custom_optimizer = Adam(learning_rate=0.001, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-07, amsgrad=False)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

# Build the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) +
              1, output_dim=100, input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

if __name__ == '__main__':
    # Compile the model
    model.compile(optimizer=custom_optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32,
              validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}')

    # After training is complete, save the model
    model.save('model.h5')
