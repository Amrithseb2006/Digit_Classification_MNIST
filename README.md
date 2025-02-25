# MNIST Digit Classification using Neural Networks

## 📌 Project Overview
This project is a deep learning-based **handwritten digit classification model** using the **MNIST dataset**. The model takes grayscale images of digits (0-9) and classifies them with high accuracy. It is implemented using **TensorFlow/Keras** and trained on a **fully connected neural network (MLP) or a Convolutional Neural Network (CNN)**.

## 📂 Dataset Details
The **MNIST dataset** consists of **70,000 grayscale images** of handwritten digits, each of size **28x28 pixels**.
- **60,000 images** for training
- **10,000 images** for testing

Each image is labeled from **0 to 9**, representing the digit it contains.

## ⚙️ Model Architecture
This project includes two types of models:

### **1️⃣ Fully Connected Neural Network (MLP Model)**
- Input Layer: **Flatten** (28x28 → 784)
- Hidden Layer 1: **Dense (128 neurons, ReLU activation)**
- Output Layer: **Dense (10 neurons, Softmax activation)**

### **2️⃣ Convolutional Neural Network (CNN Model)**
- Conv2D Layer 1: **32 filters (3x3 kernel), ReLU activation**
- MaxPooling Layer 1: **2x2 pooling**
- Conv2D Layer 2: **64 filters (3x3 kernel), ReLU activation**
- MaxPooling Layer 2: **2x2 pooling**
- Flatten Layer
- Dense Layer: **128 neurons, ReLU activation**
- Output Layer: **10 neurons, Softmax activation**

## 🚀 Installation
To run this project, install the required dependencies:
```bash
pip install tensorflow numpy matplotlib
```

## 🏃‍♂️ Usage
Run the Python script to train the model:
```python
import tensorflow as tf
from tensorflow import keras

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
```

## 📊 Results
After training for **5 epochs**, the model achieves approximately **98% accuracy** on the test dataset.

## 📌 Future Improvements
- Implement **data augmentation** for better generalization
- Use **CNN with dropout layers** to reduce overfitting
- Train with **custom handwritten digit datasets**

## 📜 License
This project is licensed under the **MIT License**.

---
💡 **Contributions are welcome!** Feel free to fork, improve, and submit a pull request. 🚀

# Digit_Classification_MNIST
