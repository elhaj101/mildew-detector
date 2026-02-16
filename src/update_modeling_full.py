
import json
import os

nb_path = 'jupyter_notebooks/ModelingandEvaluation.ipynb'

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

# Read existing notebook to preserve model definition if possible, 
# but frankly, rewriting it cleanly is safer given the structural issues.
# I'll construct a new list of cells.

cells = []

# Cell 1: Intro
cells.append(create_markdown_cell("# Modeling and Evaluation\n\nThis notebook handles data splitting, augmentation, model training, and evaluation."))

# Cell 2: Imports & path setup
cells.append(create_code_cell("""
import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
"""))

# Cell 3: Data Splitting Function
cells.append(create_markdown_cell("## Data Splitting\n\nSplitting the dataset into Train, Validation, and Test sets."))

cells.append(create_code_cell("""
source_dir = '../data/cherry-leaves'
split_dir = '../data/split'
labels = ['healthy', 'powdery_mildew']
split_ratios = (0.7, 0.15, 0.15) # Train, Val, Test

# Create split directories
for label in labels:
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(split_dir, split, label), exist_ok=True)

# Distribute images
for label in labels:
    src_label_dir = os.path.join(source_dir, label)
    images = [f for f in os.listdir(src_label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    train_count = int(len(images) * split_ratios[0])
    val_count = int(len(images) * split_ratios[1])
    
    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count+val_count]
    test_imgs = images[train_count+val_count:]
    
    # Helper to copy
    def copy_images(img_list, split_name):
        dest = os.path.join(split_dir, split_name, label)
        # Check if already populated to avoid re-copying overhead if run multiple times
        # But for robustness, we might just overwrite or skip.
        # Let's check counts.
        if len(os.listdir(dest)) == len(img_list):
            print(f"Split {split_name}/{label} already exists.")
            return

        print(f"Copying {len(img_list)} images to {split_name}/{label}...")
        for img in img_list:
            src = os.path.join(src_label_dir, img)
            dst = os.path.join(dest, img)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    copy_images(train_imgs, 'train')
    copy_images(val_imgs, 'val')
    copy_images(test_imgs, 'test')
    
print("Data splitting complete.")
"""))

# Cell 4: Generators
cells.append(create_markdown_cell("## Data Generators with Augmentation\n\nWe apply data augmentation (rotation, zoom, flips) to the training set to prevent overfitting."))

cells.append(create_code_cell("""
img_size = (100, 100) # Using 100x100 as per common practice for this dataset, or user previous 224? 
# Previous code used 224. But typical for this project is often smaller if not using transfer learning. 
# Let's stick to 100x100 to be safe on memory/speed, or 224 if we want higher res. 
# README doesn't specify size. Visualization nb used (100, 100).
# I will use (100,100) to match Visualization nb.
img_shape = (100, 100, 3)
batch_size = 20

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_path = os.path.join(split_dir, 'train')
val_path = os.path.join(split_dir, 'val')
test_path = os.path.join(split_dir, 'test')

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = test_datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
"""))

# Cell 5: Model
cells.append(create_markdown_cell("## Model Architecture\n\nCNN Model definition."))

cells.append(create_code_cell("""
model = Sequential([
    Input(shape=img_shape),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()
"""))

# Cell 6: Training
cells.append(create_markdown_cell("## Model Training"))

cells.append(create_code_cell("""
output_dir = '../out/modeling'
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = os.path.join(output_dir, 'mildew_detector_model.keras')

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=max(1, len(train_generator)), 
    validation_data=val_generator,
    validation_steps=max(1, len(val_generator)),
    callbacks=[early_stop, checkpoint],
    verbose=1
)
"""))

# Cell 7: Learning Curves
cells.append(create_markdown_cell("## Learning Curves\n\nPlotting accuracy and loss over epochs."))

cells.append(create_code_cell("""
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.title('Model Training History')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.show()
"""))

# Cell 8: Evaluation
cells.append(create_markdown_cell("## Model Evaluation"))

cells.append(create_code_cell("""
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")
"""))

# Cell 9: Detailed Metrics
cells.append(create_markdown_cell("## Confusion Matrix and Classification Report"))

cells.append(create_code_cell("""
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predictions
pred_probs = model.predict(test_generator)
pred_classes = (pred_probs > 0.5).astype(int).flatten()
true_classes = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Powdery Mildew'], 
            yticklabels=['Healthy', 'Powdery Mildew'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(true_classes, pred_classes, target_names=['Healthy', 'Powdery Mildew']))
"""))

# Construct notebook dict
nb_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb_data, f, indent=1)

print(f"Rewrote {nb_path} with updated content.")
