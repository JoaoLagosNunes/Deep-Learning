import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import urllib
import time
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np


# Function to search for images using DuckDuckGo with Selenium
def search_images(query, category, num_images):
    driver = webdriver.Chrome()  # You need to have Chrome WebDriver installed
    driver.get(f"https://duckduckgo.com/?q={query}&iax=images&ia=images")
    time.sleep(2)  # Wait for the page to load (you might need to adjust the wait time)

    img_elements = driver.find_elements(By.CSS_SELECTOR, ".tile--img__img.js-lazyload")
    img_urls = [element.get_attribute("src") for element in img_elements]

    driver.quit()

    os.makedirs(os.path.join("img", category), exist_ok=True)
    for i, img_url in enumerate(img_urls[:num_images], start=1):
        img_name = f"{category}_{i}.jpg"
        img_path = os.path.join("img", category, img_name)
        try:
            urllib.request.urlretrieve(img_url, img_path)
            print(f"Downloaded: {img_name}")
        except Exception as e:
            print(f"Error downloading image - {e}")


# Function to check image quality and dimensions
def check_image_quality(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            quality = img.info.get(
                "quality", 95
            )  # Default to 95 if quality info is not available
            return (
                width >= min_width and height >= min_height and quality >= min_quality
            )
    except Exception as e:
        print(f"Error checking image quality: {e}")
        return False


# Streamlit UI
st.title("Image Classification Streamlit App")

# Task 1: Image Scraping
if st.button("Scrape Images"):
    st.text("Scraping images...")
    categories = {"car": 100, "bike": 100, "motorbike": 100, "bus": 100, "train": 100}
    for category, num_images in categories.items():
        st.text(f"Downloading images for {category}...")
        search_images(category, category, num_images)
    st.success("Image scraping completed.")

# Task 2: Image Quality Check
if st.button("Check Image Quality"):
    st.text("Checking image quality...")
    data_dir = "img"
    categories = ["car", "bike", "motorbike", "bus", "train"]
    min_width, min_height = 100, 100
    min_quality = 90
    for category in categories:
        category_path = os.path.join(data_dir, category)
        filtered_images = []
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            if check_image_quality(img_path):
                filtered_images.append(img_name)
            else:
                os.remove(img_path)
        st.text(
            f"Filtered {len(os.listdir(category_path)) - len(filtered_images)} images in '{category}' category."
        )
    st.success("Image quality check completed.")

# Task 3: Data Loading and Augmentation
data_dir = "img"
categories = {"car": 100, "bike": 100, "motorbike": 100, "bus": 100, "train": 100}
if st.button("Load and Augment Data and Train Model"):
    st.text("Loading and augmenting data...")
    batch_size = 32
    image_size = (150, 150)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )
    st.success("Data loading and augmentation completed.")

    st.text("Training the model...")
    num_classes = len(categories)
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=1e-7
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    callbacks = [reduce_lr, early_stopping]
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    st.success("Model training completed.")

# Task 5: Model Evaluation
if st.button("Evaluate Model"):
    st.text("Evaluating the model...")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        "img",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_pred_probs = model.predict(test_generator)
    test_pred_labels = np.argmax(test_pred_probs, axis=1)
    true_labels = test_generator.classes
    st.text("Classification Report:")
    st.text(
        classification_report(true_labels, test_pred_labels, target_names=categories)
    )

    confusion_mtx = confusion_matrix(true_labels, test_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_mtx,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot()
