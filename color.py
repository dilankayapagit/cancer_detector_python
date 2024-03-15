import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_colors_and_masks(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper HSV color thresholds for skin lesions (adjust these values as needed)
    lower_bound = np.array([1, 1, 1])
    upper_bound = np.array([210, 210, 210])

    # Create a mask that isolates the potential cancerous regions
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Bitwise AND to separate the potential cancerous regions from the original image
    cancerous_skin = cv2.bitwise_and(image, image, mask=mask)

    # Convert the cancerous skin image to grayscale
    gray_cancerous_skin = cv2.cvtColor(cancerous_skin, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask of the skin lesion
    _, binary_mask = cv2.threshold(gray_cancerous_skin, 1, 255, cv2.THRESH_BINARY)

    # Find connected components in the binary mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Initialize lists to store masks for moles with one color and more than one color
    one_color_masks = []
    multi_color_masks = []

    # Loop through the connected components (ignore the first component, which is the background)
    for label in range(1, num_labels):
        # Create a mask for the current connected component
        current_mask = (labels == label).astype(np.uint8) * 255

        # Calculate the unique colors in the current mask
        unique_colors = np.unique(cancerous_skin[current_mask > 0], axis=0)

        # Determine whether there is one or more than one color
        if len(unique_colors) == 1:
            one_color_masks.append(current_mask)
        else:
            multi_color_masks.append(current_mask)

    # Display the image with color information
    plt.imshow(cv2.cvtColor(cancerous_skin, cv2.COLOR_BGR2RGB))
    plt.title(f"Color Info: {len(one_color_masks)} moles with one color, {len(multi_color_masks)} moles with more than one color")
    plt.axis('off')
    plt.show()

    # Return the masks for moles with one color and more than one color
    return one_color_masks, multi_color_masks

image_paths = [
    r"img_edit/ISIC_0000142.jpg",
    r"img_edit/ISIC_0000145.jpg",
    r"img_edit/ISIC_0000166.jpg",
]

for image_path in image_paths:
    one_color_masks, multi_color_masks = analyze_colors_and_masks(image_path)

    # You can further process the masks as needed
    for mask in one_color_masks:
        # Process masks with one color
        pass

    for mask in multi_color_masks:
        # Process masks with more than one color
        pass
