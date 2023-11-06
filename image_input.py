from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('Model/image_model_v1.h5')

# Load and preprocess the new image
new_image_path = 'Datasets/dataset/other_images/dog.jpg'
new_image = image.load_img(new_image_path, target_size=(64, 64))
new_image = image.img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)
new_image = new_image / 255.0  # Normalize pixel values

prediction = model.predict(new_image)

# Display the new image
plt.imshow(new_image[0])
plt.axis('off')

# Interpret the prediction
if prediction < 0.5:
    prediction_text = 'Predicted: Cat'
else:
    prediction_text = 'Predicted: Dog'

# Write the prediction text on the plot
plt.text(10, 10, prediction_text, color='white',
         backgroundcolor='black', fontsize=10)

plt.show()
