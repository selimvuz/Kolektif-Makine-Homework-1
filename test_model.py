from keras.models import load_model
import pandas as pd

# Load the test data
X_test = pd.read_csv('X_test.csv')

model = load_model('model.h5')

y_pred = model.predict(X_test)

evaluations = model.evaluate(X_test, y_pred)

# Print the evaluation metrics
print(f"Loss: {evaluations[0]}")
print(f"Accuracy: {evaluations[1]}")
