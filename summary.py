from model import model
from keras.models import load_model

model = load_model('Model/model_adam.h5')

summary = model.summary()

print(summary)
