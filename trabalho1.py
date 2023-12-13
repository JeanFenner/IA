'''
Universidade Federal da Fronteira Sul – UFFS
Ciência da Computação – Inteligência Artificial
Profº Felipe Grando
Jean Carlo Fenner

Aprendizado Não Supervisionado com o Algoritmo K-Médias
'''
import cv2
import numpy as np
from sklearn.cluster import KMeans

def reduce_colors(image_path, k):
    # Original image
    image = cv2.imread(image_path)    
    height, width, _ = image.shape
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    
    # Map the labels to the centers
    new_image = centers[labels.flatten()]
    
    # Reshape back to the original image shape
    new_image = new_image.reshape(image.shape)
    
    return new_image, height, width

# Example usage
input_image_path = "CC\IA\img.png"

# Load the original image
original_image = cv2.imread(input_image_path)

# Process the image
new_image, img_height, img_width = reduce_colors(input_image_path, 2)

# Print details of the original image
print("Image Details:")
print(f"Resolution: {img_width}x{img_height}")
print(f"Number of Colors: {len(np.unique(original_image.reshape(-1, original_image.shape[2]), axis=0))}")

# Save the result
output_image_path = "CC\IA\img_n.png"
cv2.imwrite(output_image_path, new_image)