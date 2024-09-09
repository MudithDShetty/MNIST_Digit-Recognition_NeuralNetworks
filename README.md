# MNIST_Digit-Recognition_NeuralNetworks
### MNIST Digit Classification Model with Image Augmentation and Custom Image Prediction

This script builds a neural network model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model includes image augmentation, dropout layers to prevent overfitting, and custom functionality to predict handwritten digits from images. Below is a detailed documentation of the code:

---

### 1. **Loading and Preprocessing the MNIST Dataset**

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9) that are 28x28 pixels each.

```python
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
```

- **Dataset**: Split into training (60,000 images) and test sets (10,000 images).
- **Normalization**: Pixel values are scaled between 0 and 1 to aid the model's learning process.

---

### 2. **Data Augmentation**

To improve the generalization of the model, random transformations are applied to the training images using `ImageDataGenerator`.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,        # Random rotations within 10 degrees
    width_shift_range=0.1,    # Random horizontal shifts
    height_shift_range=0.1,   # Random vertical shifts
    zoom_range=0.1            # Random zoom within 10%
)
datagen.fit(train_images.reshape(-1, 28, 28, 1))
```

---

### 3. **Building the Model**

A Sequential neural network model is built with the following architecture:
- **Input Layer**: Flatten 28x28 images into 1D vectors.
- **Dense Layers**: Two fully connected layers with 256 neurons each and ReLU activation.
- **Dropout Layers**: To prevent overfitting, dropout layers with 30% probability are added after each dense layer.
- **Output Layer**: A softmax layer for classification into 10 classes (digits 0-9).

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])
```

---

### 4. **Model Compilation and Training**

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metrics. The model is trained on the augmented images for 20 epochs, with a batch size of 32.

```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=32),
                    epochs=20, 
                    validation_data=(test_images, test_labels))
```

---

### 5. **Evaluating the Model**

After training, the model is evaluated on the test dataset to obtain the final accuracy.

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
```

---

### 6. **Making Predictions on the Test Set**

To visualize the model's predictions, the `plot_image` and `plot_value_array` functions are used. These functions display the input images along with predicted and actual labels.

```python
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    plt.xlabel(f"{predicted_label} ({100*np.max(predictions_array):.2f}%)")

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.bar(range(10), predictions_array)
    plt.ylim([0, 1])
    plt.xticks(range(10))
```

---

### 7. **Loading and Predicting Custom Images**

This section handles loading custom grayscale images of handwritten digits, resizing them to 28x28 pixels, and preparing them for prediction by the model. A `display_prediction` function is used to show the image along with the predicted digit.

- **Inverting Colors**: The image is inverted to match the MNIST data format where black is the background and white represents the digits.

```python
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def display_prediction(image_path):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    img_display = Image.open(image_path)
    plt.imshow(img_display, cmap=plt.cm.gray)
    plt.title(f"Predicted Digit: {predicted_label}")
    plt.axis('off')
    plt.show()
```


---

### 8. **Sample Usage for Custom Image Prediction**

The following calls demonstrate how to predict digits from custom images saved at the specified file paths.

```python
image_path = "C:/Users/ASUS/Downloads/2.jpg"  
display_prediction(image_path)

image_path = "C:/Users/ASUS/Downloads/8.jpg"  
display_prediction(image_path)

image_path = "C:/Users/ASUS/Downloads/0.jpg"  
display_prediction(image_path)
```

---

### Conclusion

This documentation outlines the process of building an MNIST digit classifier using Keras, with augmentation techniques to improve model generalization. The model can predict digits from both the MNIST dataset and custom input images by preprocessing and resizing them to match the input format of the model.
