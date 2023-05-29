import os
import numpy as np
from PIL import Image


def combine_images(original_images_path, generated_masks_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Iterate over the original images
    for filename in os.listdir(original_images_path):
        if filename.endswith(".jpg"):
            # Load original image
            image_path = os.path.join(original_images_path, filename)
            original_image = Image.open(image_path)

            # Load generated mask
            mask_path = os.path.join(generated_masks_path, filename)
            mask_image = Image.open(mask_path).convert("L")

            # Resize mask image to match original image dimensions
            mask_image = mask_image.resize(original_image.size, Image.NEAREST)

            # Convert mask to RGB format
            mask_image = mask_image.convert("RGB")
            mask_pixels = np.array(mask_image)

            # Set segmented region to a color (e.g., green)
            mask_pixels[mask_pixels.any(axis=2)] = [0, 255, 0]

            # Combine original image and mask
            combined_image = Image.blend(original_image, mask_image, alpha=0.5)

            # Save the combined image
            output_filename = os.path.join(output_path, filename)
            combined_image.save(output_filename)



original_images_path = "/mnt/Depolama/BMCOURSES/4th_term/BM686/assignments/Final/Kvasir-SEG/test/img"
generated_masks_path = "/mnt/Depolama/BMCOURSES/4th_term/BM686/assignments/Final/Kvasir-SEG/test/labelcol"
output_path = "/mnt/Depolama/BMCOURSES/4th_term/BM686/assignments/Final/Kvasir-SEG/test/blended_images_with_masks "

combine_images(original_images_path, generated_masks_path, output_path)
