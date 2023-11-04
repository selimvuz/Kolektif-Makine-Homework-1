from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

# Assuming you saved your model as 'model_CCE.h5'
model = load_model('model_v3.h5')

# Define an example input
example_input = ["I hate you! Get lost! You are disgusting!"]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(example_input)

# Preprocess the input (tokenize, etc.)
tokenized_input = tokenizer.texts_to_sequences(example_input)
padded_sequence = pad_sequences(
    tokenized_input, maxlen=100, padding='post', truncating='post')

# Make a prediction
predicted_output = model.predict(padded_sequence)

print("Value: ", predicted_output)

# Get the index of the highest probability
predicted_class = predicted_output.argmax()

# Define a list of sentiment labels corresponding to the classes
sentiment_labels = ['Pozitif', 'Nötr', 'Negatif']

# Print out the predicted sentiment
print(f'Tahmin Edilen Duygu: {sentiment_labels[predicted_class]}')
