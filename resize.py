import os
from PIL import Image

def resize_images_in_directory(directory, size):
    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.jpg')):
                # Construct the file path
                file_path = os.path.join(root, file)

                # Open the image and resize it
                image = Image.open(file_path)
                resized_image = image.resize(size)

                # Replace the original image with the resized image
                resized_image.save(file_path)
                print(f"Resized image saved: {file_path}")

# Example usage
directory = '/mnt/Depolama/BMCOURSES/4th_term/BM686/assignments/Final/Kvasir-SEG'
image_size = (128, 128)  # New size of the images

resize_images_in_directory(directory, image_size)


