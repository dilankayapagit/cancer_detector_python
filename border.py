import cv2
import numpy as np
import matplotlib.pyplot as plt

def diameter(path):
    # Load the image
    image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper HSV color thresholds for skin lesions (adjust these values as needed)
    lower_bound = np.array([1, 1, 1])
    upper_bound = np.array([210,210, 210])

    # Create a mask that isolates the potential cancerous regions
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store the largest contour and its perimeter
    largest_contour = None
    max_perimeter = 0

    # Iterate through the contours and find the largest contour
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            largest_contour = contour

    # Calculate the approximate polygonal representation of the contour
    epsilon = 0.02 * max_perimeter
    approximated_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Check the number of vertices in the approximated contour
    num_vertices = len(approximated_contour)

    # Determine whether the border is even based on the number of vertices
    even_border = True if num_vertices >= 7 else False

    # Draw the largest contour on the image
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)

    # Annotate the image with the evenness of the border
    border_text = 'Even Border' if even_border else 'Irregular Border'
    cv2.putText(image, border_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR image to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using pyplot
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()




image_paths = [
    r"img_edit/ISIC_0000142.jpg",
    r"img_edit/ISIC_0000145.jpg",
    r"img_edit/ISIC_0000166.jpg",
]

for image_path in image_paths:
    diameter(image_path)