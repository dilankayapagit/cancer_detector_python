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

    # Bitwise AND to separate the potential cancerous regions from the original image
    # cancerous_skin = cv2.bitwise_and(image, image, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the largest diameter and its corresponding contour
    max_diameter = 0
    largest_contour = None

    # Iterate through the contours and find the largest diameter
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter = int(2 * radius)
        
        if diameter > max_diameter:
            max_diameter = diameter
            largest_contour = contour

    # Draw the largest contour on the image
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)

    # Annotate the image with the diameter
    diameter_text = f'Diameter: {round(max_diameter/90,1)} mm'
    cv2.putText(image, diameter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR image to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using pyplot
    
    plt.figure(figsize=(5, 5))
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