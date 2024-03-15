import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_compare_masks(image_path1, image_path2):
    # Load the two images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert the images to the HSV color space
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Define lower and upper HSV color thresholds for skin lesions (adjust these values as needed)
    lower_bound = np.array([1, 1, 1])
    upper_bound = np.array([210, 210, 210])

    # Create masks that isolate potential cancerous regions in both images
    mask1 = cv2.inRange(hsv_image1, lower_bound, upper_bound)
    mask2 = cv2.inRange(hsv_image2, lower_bound, upper_bound)

    # Find connected components in the binary masks
    _, labels1, _, _ = cv2.connectedComponentsWithStats(mask1, connectivity=8)
    _, labels2, _, _ = cv2.connectedComponentsWithStats(mask2, connectivity=8)

    # Calculate the number of connected components (potential cancer cells) in each image
    num_cells1 = np.max(labels1)
    num_cells2 = np.max(labels2)

    # Display the images with masks
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Image 1: {num_cells1} potential cancer cells")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Image 2: {num_cells2} potential cancer cells")
    axes[1].axis('off')

    plt.show()

    # Compare the number of potential cancer cells between the two images
    if num_cells2 > num_cells1:
        print("The cancer cells are evolving (increasing in number).")
    elif num_cells2 < num_cells1:
        print("The cancer cells are regressing (decreasing in number).")
    else:
        print("The number of cancer cells remains unchanged.")

# Example usage with two different images
image_path1 = r"img_edit\s1.jpg"
image_path2 = r"img_edit\e1.jpg"
analyze_and_compare_masks(image_path1, image_path2)
