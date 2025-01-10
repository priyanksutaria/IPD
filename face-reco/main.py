import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt

# Update my_label function for 3 people: Priyank, Sujal, Sneh
def my_label(image_name):
    name = image_name.split('.')[-3] 
    if name == "Priyank":
        return np.array([1, 0, 0])
    elif name == "Sujal":
        return np.array([0, 1, 0])
    elif name == "Sneh":
        return np.array([0, 0, 1])

# Update my_data function to handle the 3 people scenario
def my_data():
    data = []
    for img in tqdm(os.listdir("data")):
        path = os.path.join("data", img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)  
    return data

# Load data
data = my_data()
train = data[:2500]
test = data[2500:]

X_train = np.array([i[0] for i in train]).reshape(-1, 50, 50, 1)
y_train = np.array([i[1] for i in train])
X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
y_test = np.array([i[1] for i in test])

# Path where the model will be saved
model_path = "face_recognition_model.h5"

# Check if the model file already exists
if os.path.exists(model_path):
    print("Loading saved model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("Training a new model...")
    model = models.Sequential()
    model.add(layers.Input(shape=(50, 50, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))  # 3 classes now

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test), verbose=1)

    # Save the model to the specified path
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Visualization function to handle 3 labels
def data_for_visualization():
    Vdata = []
    for img in tqdm(os.listdir("Images for visualization")):
        path = os.path.join("Images for visualization", img)
        img_num = img.split('.')[0] 
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        Vdata.append([np.array(img_data), img_num])
    shuffle(Vdata)
    return Vdata

Vdata = data_for_visualization()

# Visualization
fig = plt.figure(figsize=(15, 12))  # Adjusted the figure size for more images
rows = 6  # Increase rows to 6 to accommodate 30 images
cols = 5  # Keeping columns as 5 for better layout

# Plot all 30 images
for num, data in enumerate(Vdata[:30]):  # Show all 30 images
    img_data = data[0]
    y = fig.add_subplot(rows, cols, num+1)  # Adjusted grid size
    image = img_data
    data = img_data.reshape(1, 50, 50, 1)
    model_out = model.predict(data)[0]
    
    # Find the predicted label
    if np.argmax(model_out) == 0:
        label = 'Priyank'
    elif np.argmax(model_out) == 1:
        label = 'Sujal'
    else:
        label = 'Sneh'
    
    y.imshow(image, cmap='gray')
    plt.title(label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()
