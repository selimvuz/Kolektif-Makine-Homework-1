from keras.preprocessing.sequence import pad_sequences
from model import X_test, y_test
from keras.models import load_model
import pandas as pd

model = load_model('model_v8.h5')

X_test = pad_sequences(X_test, maxlen=100, truncating='pre',
                       padding='pre', value=0)

predicts = model.predict(X_test)

# Get the predicted classes (assuming a classification task)
predicted_classes = predicts.argmax(axis=1)

df = pd.DataFrame({'Tahminler': predicted_classes,
                  'Gerçek Değerler': y_test})

# Save the predictions to a CSV file
df.to_csv('predictions_table.csv', index=False)
