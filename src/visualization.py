import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
import cv2
import os

def show_sample_images(generator, label_map, num_images=6, title="Sample Images"):
    class_names = {v: k for k, v in label_map.items()}
    images, labels = next(generator)
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {class_names[int(labels[i])]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def apply_gradcam(model, img_array, layer_name):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Binary classification

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    # Get original image dimensions
    img_height, img_width = img.shape[:2]
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    
    # Convert heatmap to color map
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Convert original image to proper format for OpenCV
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Ensure both images have the same dimensions and channels
    if img_bgr.shape != heatmap_colored.shape:
        print(f"Image shape: {img_bgr.shape}, Heatmap shape: {heatmap_colored.shape}")
        # If there's still a mismatch, resize heatmap again
        heatmap_colored = cv2.resize(heatmap_colored, (img_width, img_height))
    
    # Create overlay
    overlay = cv2.addWeighted(img_bgr, 1.0, heatmap_colored, alpha, 0)
    
    # Convert back to RGB for matplotlib display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb

def show_predictions_with_gradcam(model, generator, layer_name, num_images=5):
    images, labels = next(generator)
    preds = model.predict(images)

    plt.figure(figsize=(15, 3 * num_images))
    for i in range(num_images):
        img = images[i]
        img_input = np.expand_dims(img, axis=0)

        heatmap = apply_gradcam(model, img_input, layer_name)
        overlay = overlay_gradcam(img, heatmap)

        pred_label = int(preds[i] > 0.5)
        true_label = int(labels[i])

        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title(f"Original\nTrue: {true_label} | Pred: {pred_label}")
        plt.axis("off")

        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_pneumonia_gradcam(model, generator, layer_name, num_images=5):
    """
    Find pneumonia cases and show Grad-CAM visualizations
    """
    pneumonia_images = []
    pneumonia_labels = []
    normal_images = []
    normal_labels = []
    
    # Collect samples until we have enough pneumonia cases
    samples_collected = 0
    max_batches = 10  # Prevent infinite loop
    batch_count = 0
    
    print("Searching for pneumonia cases...")
    
    while len(pneumonia_images) < num_images and batch_count < max_batches:
        try:
            images, labels = next(generator)
            
            for i in range(len(images)):
                if labels[i] == 1 and len(pneumonia_images) < num_images:  # Pneumonia case
                    pneumonia_images.append(images[i])
                    pneumonia_labels.append(labels[i])
                elif labels[i] == 0 and len(normal_images) < num_images:  # Normal case for comparison
                    normal_images.append(images[i])
                    normal_labels.append(labels[i])
            
            batch_count += 1
            
        except StopIteration:
            # Reset generator if we reach the end
            generator.reset()
            batch_count += 1
    
    print(f"Found {len(pneumonia_images)} pneumonia cases and {len(normal_images)} normal cases")
    
    if len(pneumonia_images) == 0:
        print("No pneumonia cases found in the current batch. Let's check class distribution...")
        return
    
    # Get predictions for pneumonia cases
    pneumonia_array = np.array(pneumonia_images)
    pneumonia_preds = model.predict(pneumonia_array)
    
    # Visualize pneumonia cases
    plt.figure(figsize=(15, 3 * len(pneumonia_images)))
    for i in range(len(pneumonia_images)):
        img = pneumonia_images[i]
        img_input = np.expand_dims(img, axis=0)

        heatmap = apply_gradcam(model, img_input, layer_name)
        overlay = overlay_gradcam(img, heatmap)

        pred_label = int(pneumonia_preds[i] > 0.5)
        true_label = int(pneumonia_labels[i])
        confidence = float(pneumonia_preds[i])

        plt.subplot(len(pneumonia_images), 2, 2 * i + 1)
        plt.imshow(img)
        plt.title(f"Original - PNEUMONIA\nTrue: {true_label} | Pred: {pred_label} | Conf: {confidence:.3f}")
        plt.axis("off")

        plt.subplot(len(pneumonia_images), 2, 2 * i + 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")

    plt.suptitle("Grad-CAM Analysis - Pneumonia Cases", fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_normal_vs_pneumonia_gradcam(model, generator, layer_name):
    """
    Compare Grad-CAM patterns between normal and pneumonia cases
    """
    pneumonia_images = []
    pneumonia_labels = []
    normal_images = []
    normal_labels = []
    
    # Collect samples
    batch_count = 0
    max_batches = 10
    
    print("Collecting normal and pneumonia cases for comparison...")
    
    while (len(pneumonia_images) < 3 or len(normal_images) < 3) and batch_count < max_batches:
        try:
            images, labels = next(generator)
            
            for i in range(len(images)):
                if labels[i] == 1 and len(pneumonia_images) < 3:  # Pneumonia
                    pneumonia_images.append(images[i])
                    pneumonia_labels.append(labels[i])
                elif labels[i] == 0 and len(normal_images) < 3:  # Normal
                    normal_images.append(images[i])
                    normal_labels.append(labels[i])
            
            batch_count += 1
            
        except StopIteration:
            generator.reset()
            batch_count += 1
    
    if len(pneumonia_images) == 0:
        print("No pneumonia cases found!")
        return
    
    # Get predictions
    all_images = normal_images + pneumonia_images
    all_labels = normal_labels + pneumonia_labels
    all_preds = model.predict(np.array(all_images))
    
    # Create comparison visualization
    plt.figure(figsize=(18, 12))
    
    # Plot normal cases
    for i in range(min(3, len(normal_images))):
        img = normal_images[i]
        img_input = np.expand_dims(img, axis=0)
        heatmap = apply_gradcam(model, img_input, layer_name)
        overlay = overlay_gradcam(img, heatmap)
        
        pred_label = int(all_preds[i] > 0.5)
        confidence = float(all_preds[i])
        
        # Original image
        plt.subplot(3, 4, i*4 + 1)
        plt.imshow(img)
        plt.title(f"NORMAL\nPred: {pred_label} | Conf: {confidence:.3f}")
        plt.axis("off")
        
        # Grad-CAM
        plt.subplot(3, 4, i*4 + 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.axis("off")
    
    # Plot pneumonia cases
    for i in range(min(3, len(pneumonia_images))):
        img = pneumonia_images[i]
        img_input = np.expand_dims(img, axis=0)
        heatmap = apply_gradcam(model, img_input, layer_name)
        overlay = overlay_gradcam(img, heatmap)
        
        pred_label = int(all_preds[len(normal_images) + i] > 0.5)
        confidence = float(all_preds[len(normal_images) + i])
        
        # Original image
        plt.subplot(3, 4, i*4 + 3)
        plt.imshow(img)
        plt.title(f"PNEUMONIA\nPred: {pred_label} | Conf: {confidence:.3f}")
        plt.axis("off")
        
        # Grad-CAM
        plt.subplot(3, 4, i*4 + 4)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.axis("off")
    
    plt.suptitle("Comparison: Normal vs Pneumonia - Grad-CAM Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()

def check_class_distribution(generator):
    """
    Check the distribution of classes in your test set
    """
    print("Checking class distribution in test set...")
    
    normal_count = 0
    pneumonia_count = 0
    total_batches = 0
    
    # Count samples in a few batches
    for _ in range(5):  # Check first 5 batches
        try:
            images, labels = next(generator)
            normal_count += np.sum(labels == 0)
            pneumonia_count += np.sum(labels == 1)
            total_batches += 1
        except StopIteration:
            break
    
    print(f"In {total_batches} batches:")
    print(f"Normal cases: {normal_count}")
    print(f"Pneumonia cases: {pneumonia_count}")
    print(f"Total: {normal_count + pneumonia_count}")
    
    if pneumonia_count > 0:
        print("Found pneumonia cases - proceeding with Grad-CAM analysis")
    else:
        print("No pneumonia cases found in the sampled batches")
    
    # Reset generator
    generator.reset()
    
    return normal_count, pneumonia_count