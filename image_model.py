from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import pandas as pd
import numpy as np

# Eğitim ve test oluşturucularını tanımla
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'Datasets/dataset/training_set',  # Path to the training set directory
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'Datasets/dataset/test_set',  # Path to the test set directory
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'Datasets/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Modeli tanımla
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification, so use 'sigmoid'
])

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Modelin özeti
model.summary()

# Modeli eğit
model.fit(
    train_generator,
    epochs=5,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Test veri kümesi üzerinde modeli değerlendir
test_loss, test_accuracy = model.evaluate(
    test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_accuracy}')

# Modeli kaydet
model.save('Model/image_model_v1.h5')
