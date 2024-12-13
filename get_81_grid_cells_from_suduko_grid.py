import cv2
import numpy as np

def remove_large_black_lines(binary_image, size_threshold=400):
    """
    Removes connected components of black pixels in a binary image if their size exceeds the given threshold.
    
    Args:
        binary_image (np.ndarray): Binary image (0 for black, 255 for white).
        size_threshold (int): The maximum allowed size for black connected components.
    
    Returns:
        np.ndarray: Binary image with large black lines removed.
    """
    
    # Invert the binary image (black pixels become white and vice versa)
    inverted_image = cv2.bitwise_not(binary_image)
    
    # Perform connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)
    
    # Create a mask to keep only small components
    mask = np.zeros_like(binary_image, dtype=np.uint8)
    
    for i in range(1, num_labels):  # Skip the background (label 0)
        if stats[i, cv2.CC_STAT_AREA] <= size_threshold:
            mask[labels == i] = 255
    
    # Invert the mask back to match the original binary convention
    result_image = cv2.bitwise_not(mask)
    return result_image

def load_and_threshold_image(image_path):
    """
    Loads an image in RGB, resizes it to 450x450, converts to grayscale, and applies adaptive thresholding.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        np.ndarray: Binary image after adaptive thresholding.
    """
    # Step 1: Load the image as RGB and resize to 450x450
    rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if rgb_image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    rgb_image = cv2.resize(rgb_image, (450, 450))           # Resize to 450x450 pixels
    
    # Step 2: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Step 3: Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(
        grayscale_image,           # Source image
        255,                       # Maximum value to assign
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method (Gaussian)
        cv2.THRESH_BINARY,         # Threshold type (binary thresholding)
        11,                        # Block size (size of neighborhood)
        2                          # Constant to subtract from mean
    )
    
    return binary_image

def extract_grid_cells(result_image, cell_size=50):
    """
    Divides a 450x450 binary image into 9x9 grid cells, each of size 50x50 pixels.
    
    Args:
        result_image (np.ndarray): Processed binary image of size 450x450.
        cell_size (int): Size of each cell (default is 50x50).
    
    Returns:
        list: List of 50x50 cell images as separate binary images.
    """
    grid_cells = []
    for row in range(0, 450, cell_size):
        for col in range(0, 450, cell_size):
            # Extract a single cell
            cell = result_image[row:row + cell_size, col:col + cell_size]
            grid_cells.append(cell)
    return grid_cells


def get_81_grid_cells_from_suduko_grid(image_path):
    # Example usage
    image_path = "test8.jpg"
    binary_image = load_and_threshold_image(image_path)
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = remove_large_black_lines(binary_image)
    cleaned_result = cv2.medianBlur(result, 5)
    cv2.imshow("Result Image", cleaned_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    grid_cells = extract_grid_cells(cleaned_result)
    for idx, cell in enumerate(grid_cells):
        cv2.imshow(f"Cell {idx}", cell)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"Extracted {len(grid_cells)} cells.")
