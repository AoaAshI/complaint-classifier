"""
Railway Complaint Classifier Model Trainer

Trains an EfficientNetB0-based image classifier on a local railway complaint dataset.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# --- Data Handling ---

def create_datasets():
    """
    Load image datasets for training and validation with normalization.
    Returns: (train_ds, val_ds, class_names)
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'data/raw/image_dataset',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/raw/image_dataset',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    class_names = train_ds.class_names

    # Normalize images: scale pixel values to [0, 1]
    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

# --- Model Construction ---

def build_model(num_classes):
    """
    Construct a scratch-trained EfficientNetB0 model for image classification.
    """
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None  # No pretraining; train from scratch
    )
    base_model.trainable = True  # Enable full model training

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Training Process ---

def train_model():
    """
    Train the railway complaint classifier from scratch, saving the best model checkpoint.
    """
    print("Training Railway Complaint Classifier from scratch...")
    print("Loading datasets...")

    # Load data
    train_ds, val_ds, class_names = create_datasets()
    print(f"Classes found: {class_names}")

    # Build model
    print("Building model...")
    model = build_model(len(class_names))
    print(f"Model built with {model.count_params():,} parameters.")

    # Ensure model save directory exists
    os.makedirs('models/saved_models', exist_ok=True)

    # Callbacks for early stopping and checkpointing
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/saved_models/railway_complaint_classifier.h5',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]

    # Model training
    print("Starting training (20 epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # Results and saving
    best_acc = max(history.history['val_accuracy'])
    print("\nTraining complete.")       # Only milestone emoji kept on completion
    print(f"Best validation accuracy: {best_acc:.4f}")
    print("Model saved: models/saved_models/railway_complaint_classifier.h5")

if __name__ == '__main__':
    train_model()
