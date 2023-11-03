from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from model import model

# Define an example input
example_input = "I hope everyone a wonderful day! :) #positive #smile #happy #friends #fun #laugh #play #goodtimes #goodvibes #goodday #lovely #beautiful #smile #fun #friends #love #happy #goodtimes #goodvibes #goodday #lovely #beautiful"

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(example_input)

# Assuming 'model' is your trained neural network

# Preprocess the input (tokenize, vectorize, etc.)
# This step depends on how you preprocessed your training data
# For example, using the tokenizer from the previous example:
tokenized_input = tokenizer.texts_to_sequences(example_input)
padded_sequence = pad_sequences(
    tokenized_input, maxlen=100, padding='post', truncating='post')
# input_vector = tokenizer.sequences_to_matrix(padded_sequence, mode='binary')

# Make a prediction
predicted_output = model.predict(padded_sequence)

print("Value: ", predicted_output[0])

if predicted_output[0] > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")

# The predicted_output will depend on the specific task and model architecture
# For example, in sentiment analysis, it might be a probability of positive/negative sentiment

# If needed, you can interpret the predicted output based on your task
# For example, in sentiment analysis, you might check if the probability is above a threshold for a positive sentiment
