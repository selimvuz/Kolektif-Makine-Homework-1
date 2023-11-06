from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from model import tokenizer
import numpy as np

model = load_model('Model/model_v11.h5')

# Örnek bir girdi tanımla
example_input = [
    "Bu ürün gerçekten harika! Kaliteli malzemeler kullanılmış ve çok dayanıklı. Ayrıca tasarımı da çok şık. Kullanımı çok kolay ve işlevi mükemmel. Kesinlikle tavsiye ederim!"]
example_input_two = [
    "Bu ürün tam bir hayal kırıklığıydı. Kalitesiz malzemeler kullanılmış, hemen kırılıyor. Üstelik çok pahalı. Kesinlikle tavsiye etmiyorum, paranıza yazık!"]

# Maksimum girdi uzunluğunu tanımla
max_length = 100

# Girdiyi ön işle (tokenize et ve dizilere çevir)
tokenized_input = tokenizer.texts_to_sequences(example_input)
tokenized_input_two = tokenizer.texts_to_sequences(example_input_two)

# Dizileri eşit uzunluğa kadar doldur veya kırp
padded_sequence = pad_sequences(tokenized_input, maxlen=100, truncating='pre',
                                padding='pre', value=0)
padded_sequence_two = pad_sequences(tokenized_input_two, maxlen=100, truncating='pre',
                                    padding='pre', value=0)

# Modeli kullanarak tahmin yap
predicted_output = model.predict(padded_sequence)
predicted_output_two = model.predict(padded_sequence_two)

# Tahmin çıktısını argmax kullanarak sınıf etiketine çevir
predicted_class = predicted_output.argmax()
predicted_class_two = predicted_output_two.argmax()

# Sınıf etiketlerini tanımla
sentiment_labels = ['Nötr', 'Pozitif', 'Negatif']

# Tahmin edilen duygu değerini ekrana yazdır
print("\nMetin: ", example_input)
print("Duygu Değerleri: ", predicted_output)

print(f'\nTahmin Edilen Duygu: {sentiment_labels[predicted_class]}')

print("\nMetin: ", example_input_two)
print("Duygu Değerleri: ", predicted_output_two)

print(f'\nTahmin Edilen Duygu: {sentiment_labels[predicted_class_two]}')
