import numpy as np  
import matplotlib.pyplot as plt  
import cv2  
import os  
import tensorflow as tf  
from PIL import Image  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report  
from tqdm import tqdm  

# Define the dataset directory
image_dir = r'C:\Users\AKHUL\Downloads\tumordataset\tumordata'  

# Load image file names from 'no' and 'yes' tumor directories
no_tumor_images = os.listdir(image_dir + r'\no')  
yes_tumor_images = os.listdir(image_dir + r'\yes')  

# Initialize lists for dataset and labels
dataset = []  
label = []  
img_siz = (128, 128)  # Define image size for resizing  

# Load and process images
for image_name in tqdm(no_tumor_images, desc="No Tumor"):  
    if image_name.endswith('.jpg'):  
        image = cv2.imread(os.path.join(image_dir, 'no', image_name))
        image = cv2.resize(image, img_siz)
        dataset.append(np.array(image))
        label.append(0)  

for image_name in tqdm(yes_tumor_images, desc="Tumor"):  
    if image_name.endswith('.jpg'):  
        image = cv2.imread(os.path.join(image_dir, 'yes', image_name))
        image = cv2.resize(image, img_siz)
        dataset.append(np.array(image))
        label.append(1)  

# Convert to NumPy arrays  
dataset = np.array(dataset)  
label = np.array(label)  

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)  

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)  

# Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test)  
print(f'Accuracy: {accuracy * 100:.2f}%')  

# Save the Model
model.save(os.path.join(image_dir, 'model.h5'))

# Function to predict tumor and calculate affected percentage
def make_prediction(image_path, model):  
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (128, 128))
    image_array = np.array(image_resized) / 255.0
    input_img = np.expand_dims(image_array, axis=0)
    
    result = model.predict(input_img)[0][0]
    
    if result > 0.5:
        # Convert image to grayscale and apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Calculate the percentage of affected area
        total_pixels = thresholded.size
        tumor_pixels = np.sum(thresholded == 255)
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        print(f"Tumor Detected! Affected area: {tumor_percentage:.2f}%")
    else:
        print("No Tumor Detected")

# Test Predictions
make_prediction(r'C:\Users\AKHUL\Downloads\tumordataset\tumordata\yes\y50.jpg', model)  
make_prediction(r'C:\Users\AKHUL\Downloads\tumordataset\tumordata\no\no50.jpg', model)
