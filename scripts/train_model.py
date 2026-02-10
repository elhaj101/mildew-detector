
import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def train_model():
    print("Setting up paths and seeds...")
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    source_dir = 'data/cherry-leaves'
    split_dir = 'data/split'
    output_dir = 'out/modeling'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('out/visualization', exist_ok=True)

    # Data Splitting check
    if not os.path.exists(split_dir) or not os.listdir(split_dir):
        print("Performing data split...")
        labels = ['healthy', 'powdery_mildew']
        split_ratios = (0.7, 0.15, 0.15)
        
        for label in labels:
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(split_dir, split, label), exist_ok=True)
                
        for label in labels:
            src_label_dir = os.path.join(source_dir, label)
            if not os.path.exists(src_label_dir):
                 print(f"Error: Source directory {src_label_dir} not found.")
                 return

            images = [f for f in os.listdir(src_label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)
            
            p1 = int(len(images) * split_ratios[0])
            p2 = int(len(images) * (split_ratios[0] + split_ratios[1]))
            
            train_imgs = images[:p1]
            val_imgs = images[p1:p2]
            test_imgs = images[p2:]
            
            for img in train_imgs: shutil.copy(os.path.join(src_label_dir, img), os.path.join(split_dir, 'train', label, img))
            for img in val_imgs: shutil.copy(os.path.join(src_label_dir, img), os.path.join(split_dir, 'val', label, img))
            for img in test_imgs: shutil.copy(os.path.join(src_label_dir, img), os.path.join(split_dir, 'test', label, img))
            
        print("Data split complete.")
    else:
        print("Data split directory exists. Skipping split.")

    # Generators
    print("Setting up Data Generators...")
    img_size = (100, 100)
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
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(split_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = test_datagen.flow_from_directory(
        os.path.join(split_dir, 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(split_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Model Building
    print("Building Model...")
    model = Sequential([
        Input(shape=(100, 100, 3)),
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
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()
    
    # Training
    print("Starting Training...")
    checkpoint_path = os.path.join(output_dir, 'mildew_detector_model.keras')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    
    history = model.fit(
        train_generator,
        epochs=15, # Adjusted to 15 for reasonable speed/convergence
        validation_data=val_generator,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Save history plot
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title('Model Training History')
    plt.savefig('out/visualization/model_training_history.png')
    print("Saved training history plot.")
    
    # Evaluation
    print("Evaluating on Test Set...")
    loss, acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
    # Confusion Matrix
    print("Generating Confusion Matrix...")
    pred_probs = model.predict(test_generator)
    pred_classes = (pred_probs > 0.5).astype(int).flatten()
    true_classes = test_generator.classes
    
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Powdery Mildew'], 
                yticklabels=['Healthy', 'Powdery Mildew'])
    plt.title('Confusion Matrix')
    plt.savefig('out/visualization/confusion_matrix.png')
    print("Saved confusion matrix plot.")
    
    print(classification_report(true_classes, pred_classes, target_names=['Healthy', 'Powdery Mildew']))
    
    # Save model explicitly (though checkpoint already did best)
    model.save(os.path.join(output_dir, 'final_model.keras'))
    print("Model saved to out/modeling/final_model.keras")

if __name__ == "__main__":
    train_model()
