import re

text = """
Epoch 1/20
250/250 [==============================] - 19s 73ms/step - loss: 0.6651 - accuracy: 0.6231 - val_loss: 0.5893 - val_accuracy: 0.7025
Epoch 2/20
250/250 [==============================] - 17s 68ms/step - loss: 0.5511 - accuracy: 0.7231 - val_loss: 0.5634 - val_accuracy: 0.7200
Epoch 3/20
250/250 [==============================] - 17s 67ms/step - loss: 0.4993 - accuracy: 0.7636 - val_loss: 0.5504 - val_accuracy: 0.7220
Epoch 4/20
250/250 [==============================] - 17s 67ms/step - loss: 0.4436 - accuracy: 0.7976 - val_loss: 0.5578 - val_accuracy: 0.7285
Epoch 5/20
250/250 [==============================] - 17s 66ms/step - loss: 0.3970 - accuracy: 0.8229 - val_loss: 0.5620 - val_accuracy: 0.7335
Epoch 6/20
250/250 [==============================] - 17s 68ms/step - loss: 0.3581 - accuracy: 0.8414 - val_loss: 0.6143 - val_accuracy: 0.7215
Epoch 7/20
250/250 [==============================] - 17s 67ms/step - loss: 0.3206 - accuracy: 0.8570 - val_loss: 0.6139 - val_accuracy: 0.7245
Epoch 8/20
250/250 [==============================] - 17s 67ms/step - loss: 0.2839 - accuracy: 0.8780 - val_loss: 0.6786 - val_accuracy: 0.7195
Epoch 9/20
250/250 [==============================] - 17s 67ms/step - loss: 0.2478 - accuracy: 0.8971 - val_loss: 0.6854 - val_accuracy: 0.7155
Epoch 10/20
250/250 [==============================] - 17s 69ms/step - loss: 0.2112 - accuracy: 0.9181 - val_loss: 0.7982 - val_accuracy: 0.6960
Epoch 11/20
250/250 [==============================] - 17s 67ms/step - loss: 0.1808 - accuracy: 0.9320 - val_loss: 0.7945 - val_accuracy: 0.7115
Epoch 12/20
250/250 [==============================] - 17s 68ms/step - loss: 0.1427 - accuracy: 0.9515 - val_loss: 0.8459 - val_accuracy: 0.7080
Epoch 13/20
250/250 [==============================] - 17s 67ms/step - loss: 0.1150 - accuracy: 0.9630 - val_loss: 0.9100 - val_accuracy: 0.7260
Epoch 14/20
250/250 [==============================] - 17s 68ms/step - loss: 0.0832 - accuracy: 0.9765 - val_loss: 0.9791 - val_accuracy: 0.7165
Epoch 15/20
250/250 [==============================] - 17s 67ms/step - loss: 0.0715 - accuracy: 0.9795 - val_loss: 1.0591 - val_accuracy: 0.7040
Epoch 16/20
250/250 [==============================] - 17s 67ms/step - loss: 0.0530 - accuracy: 0.9879 - val_loss: 1.0733 - val_accuracy: 0.7190
Epoch 17/20
250/250 [==============================] - 17s 68ms/step - loss: 0.0400 - accuracy: 0.9915 - val_loss: 1.2154 - val_accuracy: 0.7030
Epoch 18/20
250/250 [==============================] - 17s 67ms/step - loss: 0.0327 - accuracy: 0.9939 - val_loss: 1.3439 - val_accuracy: 0.7155
Epoch 19/20
250/250 [==============================] - 17s 67ms/step - loss: 0.0275 - accuracy: 0.9952 - val_loss: 1.3193 - val_accuracy: 0.7140
Epoch 20/20
250/250 [==============================] - 17s 67ms/step - loss: 0.0183 - accuracy: 0.9981 - val_loss: 1.3711 - val_accuracy: 0.7160
"""
# Listeleri tanımla
accuracy_list = []
val_accuracy_list = []

# Metni satırlara ayır ve değerleri listelere ekle
lines = text.strip().split('\n')
for line in lines:
    # Regex ile accuracy ve val_accuracy değerlerini bul
    match = re.search(r'accuracy: (\d+\.\d+).*val_accuracy: (\d+\.\d+)', line)
    if match:
        accuracy_list.append(float(match.group(1)))
        val_accuracy_list.append(float(match.group(2)))

# Listeleri ekrana yazdır
print("Accuracy List:", accuracy_list)
print("Validation Accuracy List:", val_accuracy_list)
