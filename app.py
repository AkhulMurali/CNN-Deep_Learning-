import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# Set dataset path
image_dir = r"C:\Users\AKHUL\Downloads\tumordataset\tumordata"

# Check if dataset path exists
if not os.path.exists(image_dir):
    st.error(f"Error: Dataset path '{image_dir}' not found.")
    st.stop()

# Load image file names
no_tumor_images = os.listdir(os.path.join(image_dir, "no"))
yes_tumor_images = os.listdir(os.path.join(image_dir, "yes"))

# Streamlit app title
st.title("Brain Tumor Detection and Percentage Estimation using CNN")
st.write(f"No Tumor images: {len(no_tumor_images)}")
st.write(f"Tumor images: {len(yes_tumor_images)}")

# Initialize dataset and labels
dataset, label = [], []
img_siz = (128, 128)

st.write("Loading dataset...")

# Load No Tumor images
for image_name in tqdm(no_tumor_images, desc="Loading No Tumor Images"):
    if image_name.endswith('.jpg'):
        image = cv2.imread(os.path.join(image_dir, "no", image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_siz)
        dataset.append(image)
        label.append(0)

# Load Tumor images
for image_name in tqdm(yes_tumor_images, desc="Loading Tumor Images"):
    if image_name.endswith('.jpg'):
        image = cv2.imread(os.path.join(image_dir, "yes", image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_siz)
        dataset.append(image)
        label.append(1)

# Convert dataset and labels to NumPy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
st.write("Training the model... Please wait.")
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

# Display training accuracy and loss plots
st.write("Training Accuracy and Loss")

# Plot Accuracy
fig, ax = plt.subplots()
ax.plot(history.epoch, history.history['accuracy'], label='Training Accuracy')
ax.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
st.pyplot(fig)

# Plot Loss
fig, ax = plt.subplots()
ax.plot(history.epoch, history.history['loss'], label='Training Loss')
ax.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
st.pyplot(fig)

# Model evaluation
st.write("Evaluating Model...")
loss, accuracy = model.evaluate(x_test, y_test)
st.write(f"Model Accuracy: {round(accuracy * 100, 2)}%")

# Image Upload and Prediction
st.write("Upload an Image for Tumor Detection and Percentage Calculation")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    """ Preprocess image for prediction """
    img = Image.open(img)
    img = img.resize((128, 128))  # Resize
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model
    return img

def calculate_tumor_percentage(img):
    """ Calculate the percentage of the tumor area """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    total_pixels = gray.size
    tumor_pixels = np.count_nonzero(thresholded)
    percentage = (tumor_pixels / total_pixels) * 100
    return round(percentage, 2)

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    processed_img = preprocess_image(uploaded_file)
    prediction = model.predict(processed_img)[0][0]
    
    result_text = "Tumor Detected" if prediction > 0.5 else "No Tumor"
    st.write(f"Prediction: **{result_text}**")
    
    if prediction > 0.5:
        img_array = np.array(Image.open(uploaded_file).resize((128, 128)))
        tumor_percentage = calculate_tumor_percentage(img_array)
        st.write(f"Estimated Tumor Coverage: **{tumor_percentage}%**")

# Save Model
model.save(os.path.join(image_dir, "model.h5"))
st.success("Model saved successfully!")
