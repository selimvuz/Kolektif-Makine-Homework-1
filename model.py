from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Veri kümesini yükle
df = pd.read_csv(
    'Datasets/TRdata.csv', encoding='utf-16').dropna(subset=['Metinler', 'Duygular'])

X = df['Metinler'].values
y = df['Duygular'].values

# Veriyi tokenleştir
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)
# Sabit bir girdi uzunluğu belirle
X = pad_sequences(X, maxlen=200, truncating='pre', padding='pre', value=0)

# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

num_classes = 3  # Sınıf sayısı

# One-hot encode
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# np.savetxt('X_test.csv', X_test, delimiter=',')

# Öğrenme oranını tanımla
learning_rate = 0.001

# Adam optimizasyonu
adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9,
                      beta_2=0.999, epsilon=1e-07, amsgrad=False)

# SGD optimizasyonu
sgd_optimizer = SGD(learning_rate=learning_rate)

# SGD + Momentum optimizasyonu
sgdm_optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

# Validation loss 3 devirde gelişmezse eğitimi durdur
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

input_dim = len(tokenizer.word_index) + 1

# Modeli tanımla
model = Sequential([
    Embedding(input_dim=input_dim, output_dim=100,
              input_length=200, mask_zero=True, embeddings_initializer=Constant(value=0.25)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(3, activation='softmax')
])

if __name__ == '__main__':
    # Modeli derle
    model.compile(optimizer=sgdm_optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Modelin eğit
    model.fit(X_train, y_train_one_hot, epochs=50, batch_size=64,
              validation_data=(X_test, y_test_one_hot), callbacks=[early_stopping])

    # Modeli hesapla
    test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
    print(f'Test accuracy: {test_accuracy}')

    # Modeli kaydet
    model.save('Model/model_sgdm_v5.h5')
