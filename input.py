from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from model import tokenizer
import numpy as np

model = load_model('Model/model_v8.h5')

# Define an example input
example_input = [
    "Bu ürün gerçekten harika! Kaliteli malzemeler kullanılmış ve çok dayanıklı. Ayrıca tasarımı da çok şık. Kullanımı çok kolay ve işlevi mükemmel. Kesinlikle tavsiye ederim!"]
example_input_two = [
    "Bu ürün tam bir hayal kırıklığıydı. Kalitesiz malzemeler kullanılmış, hemen kırılıyor. Üstelik çok pahalı. Kesinlikle tavsiye etmiyorum, paranıza yazık!"]

# Define the maximum length
max_length = 100

# Preprocess the input (tokenize, etc.)
tokenized_input = tokenizer.texts_to_sequences(example_input)
tokenized_input_two = tokenizer.texts_to_sequences(example_input_two)

# Pad the input to the appropriate length
padded_sequence = pad_sequences(tokenized_input, maxlen=100, truncating='pre',
                                padding='pre', value=0)
padded_sequence_two = pad_sequences(tokenized_input_two, maxlen=100, truncating='pre',
                                    padding='pre', value=0)

# Make a prediction
predicted_output = model.predict(padded_sequence)
predicted_output_two = model.predict(padded_sequence_two)

# Get the index of the highest probability
predicted_class = predicted_output.argmax()
predicted_class_two = predicted_output_two.argmax()

# Define a list of sentiment labels corresponding to the classes
sentiment_labels = ['Nötr', 'Pozitif', 'Negatif']

print("\nMetin: ", example_input)
print("Duygu Değerleri: ", predicted_output)

print(f'\nTahmin Edilen Duygu: {sentiment_labels[predicted_class]}')

print("\nMetin: ", example_input_two)
print("Duygu Değerleri: ", predicted_output_two)

# Print out the predicted sentiment
print(f'\nTahmin Edilen Duygu: {sentiment_labels[predicted_class_two]}')
