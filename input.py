from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from model import tokenizer
import numpy as np

model = load_model('model_v8.h5')

# Define an example input
example_input = [
    "çok güzel ve iyi bir ürün kesinlikle tavsiye ederim çok memnun kaldım çok teşekkürler"]

# Define the maximum length
max_length = 100

# Preprocess the input (tokenize, etc.)
tokenized_input = tokenizer.texts_to_sequences(example_input)

# Pad the input to the appropriate length
padded_sequence = pad_sequences(tokenized_input, maxlen=100, truncating='pre',
                                padding='pre', value=0)

# Make a prediction
predicted_output = model.predict(padded_sequence)

print("Sentiment Values: ", predicted_output)
print("Input vector: ", tokenized_input)
# print("Input vector with padding: ", padded_sequence)

# Get the index of the highest probability
predicted_class = predicted_output.argmax()

# Define a list of sentiment labels corresponding to the classes
sentiment_labels = ['Nötr', 'Pozitif', 'Negatif']

# Print out the predicted sentiment
print(f'Tahmin Edilen Duygu: {sentiment_labels[predicted_class]}')
