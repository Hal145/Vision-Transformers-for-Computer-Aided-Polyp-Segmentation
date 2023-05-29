import os
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image


def calculate_scores(original_masks_path, generated_masks_path):
    original_masks = []
    generated_masks = []

    # Load original masks
    for filename in os.listdir(original_masks_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(original_masks_path, filename)
            image = np.array(Image.open(image_path).convert("L"))
            original_masks.append(image)

    # Load generated masks
    for filename in os.listdir(generated_masks_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(generated_masks_path, filename)
            image = np.array(Image.open(image_path).convert("L"))
            generated_masks.append(image)

    # Convert lists to NumPy arrays
    original_masks = np.array(original_masks)
    generated_masks = np.array(generated_masks)

    # Flatten masks
    original_masks_flat = original_masks.flatten()
    generated_masks_flat = generated_masks.flatten()

    # Calculate Intersection over Union (IoU)
    intersection = np.logical_and(original_masks_flat, generated_masks_flat)
    union = np.logical_or(original_masks_flat, generated_masks_flat)
    iou = np.sum(intersection) / np.sum(union)

    # Calculate accuracy
    accuracy = accuracy_score(original_masks_flat, generated_masks_flat)

    # Calculate Dice coefficient (DSC)
    dsc = (2 * np.sum(intersection)) / (np.sum(original_masks_flat) + np.sum(generated_masks_flat))

    return iou, accuracy, dsc

# Example usage
original_masks_path = "/mnt/Depolama/BMCOURSES/4th_term/BM686/assignments/Final/Kvasir-SEG/test/labelcol"
generated_masks_path = "/home/hicran/Desktop/Medical-Transformer/results2/test_results"

iou_score, accuracy_score, dsc_score = calculate_scores(original_masks_path, generated_masks_path)
print("Intersection over Union (IoU):", iou_score)
print("Accuracy:", accuracy_score)
print("Dice coefficient (DSC):", dsc_score)
