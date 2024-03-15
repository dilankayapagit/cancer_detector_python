import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_asymmetric(mask):
    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return True  # Treat as asymmetric if no contours are found

    # Get the largest contour (assuming it corresponds to the skin lesion)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Split the bounding rectangle into left and right halves
    left_half = mask[y:y+h, x:x+w//2]
    right_half = mask[y:y+h, x+w//2:x+w]

    # Ensure left and right halves have the same dimensions
    min_height = min(left_half.shape[0], right_half.shape[0])
    min_width = min(left_half.shape[1], right_half.shape[1])

    left_half = left_half[:min_height, :min_width]
    right_half = right_half[:min_height, :min_width]

    # Calculate the absolute difference between the left and right halves
    diff = cv2.absdiff(left_half, right_half)
    print(diff)


    # Calculate the mean of the absolute difference
    mean_diff = np.mean(diff)

    # Define a threshold to determine if it's asymmetric or not (adjust as needed)
    asymmetry_threshold = 105  # You can adjust this threshold based on your dataset
    print(mean_diff)
    # Check if the asymmetry is above the threshold
    return mean_diff > asymmetry_threshold

def analyse_asymmetry(path):
    # Load the image
    image = cv2.imread(path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper HSV color thresholds for skin lesions (adjust these values as needed)
    lower_bound = np.array([1, 1, 1])
    upper_bound = np.array([210, 210, 210])

    # Create a mask that isolates the potential cancerous regions
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Check for asymmetry in the mask
    is_asymmetrical = is_asymmetric(mask)

    # Determine whether the mask is asymmetric or not
    asymmetry_description = "Asymmetric" if is_asymmetrical else "Symmetric"

    # Display the mask and asymmetry information
    plt.imshow(mask, cmap='gray')
    plt.title(f" {asymmetry_description}")
    plt.axis('off')
    plt.show()

# Paths to your images
image_paths = [
    r"img_edit/ISIC_0000142.jpg",
    r"img_edit/ISIC_0000145.jpg",
    # r"img_edit/ISIC_0000166.jpg",
]

for image_path in image_paths:
    analyse_asymmetry(image_path)
