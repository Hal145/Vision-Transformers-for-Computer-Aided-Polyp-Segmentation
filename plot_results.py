import os
import matplotlib.pyplot as plt


def plot_images(dir1, dir2, dir3, image_name):
    # Build the image paths for each directory

    image_path1 = os.path.join(dir1, image_name)
    image_path2 = os.path.join(dir2, image_name)
    image_path3 = os.path.join(dir3, image_name)

    # Load and plot the images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(plt.imread(image_path1))
    axes[0].set_title('Original')

    axes[1].imshow(plt.imread(image_path2))
    axes[1].set_title('Gatedaxialunet')

    axes[2].imshow(plt.imread(image_path3))
    axes[2].set_title('My model')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot image
    plt.savefig('plot_images.png')

    # Show the plot
    plt.show()

# Example usage
dir1 = '/mnt/Depolama/BMCOURSES/4th_term/BM686/assignments/Final/Kvasir-SEG/test/blended_images_with_masks'
dir2 = '/home/hicran/Desktop/Medical-Transformer/results2/blended_images_with_masks '
dir3 = '/home/hicran/Desktop/Medical-Transformer/results3/blended_images_with_masks '
image_name = 'cju0u2g7pmnux0801vkk47ivj.jpg'
# image_name = 'cju1d96gsv62d09881b3wecw2.jpg'

plot_images(dir1, dir2, dir3, image_name)
