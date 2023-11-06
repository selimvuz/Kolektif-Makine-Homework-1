from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('Model/image_model_v1.h5')

# Yeni görseli yükle ve ön işleme yap
new_image_path = 'Datasets/dataset/other_images/dog.jpg'
new_image = image.load_img(new_image_path, target_size=(64, 64))
new_image = image.img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)
new_image = new_image / 255.0  # Normalize pixel values

# Modeli kullanarak tahmin yap
prediction = model.predict(new_image)

# Grafik çiz
plt.imshow(new_image[0])
plt.axis('off')

# Tahmin değerini değişkene ata
if prediction < 0.5:
    prediction_text = 'Tahmin: Kedi'
else:
    prediction_text = 'Tahmin: Köpek'

# Tahmin değerini ekrana yazdır
plt.text(10, 10, prediction_text, color='white',
         backgroundcolor='black', fontsize=10)

# Grafikleri göster
plt.show()
