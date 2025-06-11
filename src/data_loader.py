# src/data_loader.py


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    """
    Load and preprocess data using Keras ImageDataGenerator.
    
    Args:
        data_dir (str): Path to dataset folder with train/val/test subfolders.
        img_size (tuple): Desired image size (width, height).
        batch_size (int): Number of samples per batch.
        
    Returns:
        train_generator, val_generator, test_generator
    """

    # Define paths to train, val, test
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Data augmentation for training data only
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Only rescale validation/test images (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
print("[DEBUG] data_loader.py loaded")
print("Available symbols:", dir())
