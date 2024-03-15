import cv2


def resize_image(image, target_width, target_height):
    # Get the current dimensions of the image
    height, width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = width / float(height)

    # Determine the new dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


# Example usage
image_paths = [
    r"img_edit/ISIC_0000142.jpg",
    r"img_edit/ISIC_0000145.jpg",
    r"img_edit/ISIC_0000166.jpg",
    r"img_edit/s1.jpg", 
    r"img_edit/e1.jpg", 
    r"img_edit/s2.jpg",
    r"img_edit/e2.jpg",
]


target_width = 800  # Desired width for all images
target_height = 600  # Desired height for all images

for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image
    resized_image = resize_image(image, target_width, target_height)

    # Save the resized image with a new filename or overwrite the original file
    cv2.imwrite(image_path, resized_image)

print("done")
