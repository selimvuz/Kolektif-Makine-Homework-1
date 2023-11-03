from model import model, X_test, y_test
import pandas as pd

predicts = model.predict(X_test)
df = pd.DataFrame({'Tahminler': predicts.flatten(),
                  'Gerçek Değerler': y_test})

# Save the predictions to a CSV file
df.to_csv('predictions_table.csv', index=False)
