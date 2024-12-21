import cv2
import numpy as np

# def remove_large_black_lines(binary_image, size_threshold=400):
#     """
#     Removes connected components of black pixels in a binary image if their size exceeds the given threshold.
    
#     Args:
#         binary_image (np.ndarray): Binary image (0 for black, 255 for white).
#         size_threshold (int): The maximum allowed size for black connected components.
    
#     Returns:
#         np.ndarray: Binary image with large black lines removed.
#     """
    
#     # Invert the binary image (black pixels become white and vice versa)
#     inverted_image = cv2.bitwise_not(binary_image)
    
#     # Perform connected component analysis
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)
    
#     # Create a mask to keep only small components
#     mask = np.zeros_like(binary_image, dtype=np.uint8)
    
#     for i in range(1, num_labels):  # Skip the background (label 0)
#         if stats[i, cv2.CC_STAT_AREA] <= size_threshold:
#             mask[labels == i] = 255
    
#     # Invert the mask back to match the original binary convention
#     result_image = cv2.bitwise_not(mask)
#     return result_image

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

#note if you change this pls also change the is_empty pixel count
def extract_grid_cells(result_image, cell_size=50, crop_size=40):
    """
    Divides a 450x450 binary image into 9x9 grid cells, then extracts a centered 
    cropped region (e.g., 40x40) from each 50x50 cell to avoid capturing borders.

    Args:
        result_image (np.ndarray): Processed binary image of size 450x450.
        cell_size (int): Size of each grid cell (default is 50x50).
        crop_size (int): Size of the central cropped region (default is 40x40).

    Returns:
        list: List of cropped cell images (e.g., 40x40) as separate binary images.
    """
    grid_cells = []
    margin = (cell_size - crop_size) // 2  # Margin to center the crop

    for row in range(0, 450, cell_size):
        for col in range(0, 450, cell_size):
            # Extract the full cell
            cell = result_image[row:row + cell_size, col:col + cell_size]

            # Crop the central part
            cropped_cell = cell[margin:margin + crop_size, margin:margin + crop_size]
            grid_cells.append(cropped_cell)

    return grid_cells


def estimate_noise(image):
    """
    Estimate the noise level of a grayscale image by analyzing intensity variations.
    
    Args:
        image (np.ndarray): Grayscale image.
    
    Returns:
        float: Estimated noise level (standard deviation of pixel intensities).
    """
    # Smooth the image with a Gaussian filter to remove noise
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Compute the difference between the original and smoothed images
    noise_image = cv2.absdiff(image, smoothed_image)
    
    # Calculate the standard deviation of the noise
    noise_level = np.std(noise_image)
    
    return noise_level

def adaptively_filter_image(image):
    """
    Apply a median filter with an adaptively chosen kernel size based on noise level.
    
    Args:
        image (np.ndarray): Binary image (0 for black, 255 for white).
    
    Returns:
        np.ndarray: Image after applying the median filter.
    """
    # Estimate the noise level of the image
    noise_level = estimate_noise(image)
    print(f"Estimated noise level: {noise_level}")
    
    # Determine kernel size based on noise level
    if noise_level < 25:
        return image  # Low noise, skip filtering
    elif noise_level < 30:
        ksize = 3  # Moderate noise, stronger smoothing
    else:
        ksize = 5  # High noise, aggressive smoothing
    
    print(f"Selected kernel size for median blur: {ksize}")
    
    # Apply the median filter with the chosen kernel size
    filtered_image = cv2.medianBlur(image, ksize)
    
    return filtered_image


def is_empty_cell(cell, black_pixel_threshold=400, center_weight_factor=100):
    """
    Determines if a cell is empty based on a weighted count of black pixels.
    
    Args:
        cell (np.ndarray): 50x50 binary image of a cell.
        black_pixel_threshold (int): Weighted threshold to consider the cell non-empty.
        center_weight_factor (int): Factor to multiply black pixels near the center.
    
    Returns:
        bool: True if the cell is empty, False otherwise.
    """
    size = cell.shape[0]
    half_center = size // 4  # This will give a 25x25 region for 50x50 cells
    weights = np.ones_like(cell, dtype=float)  # Default weight is 1

    # Apply a higher weight to the center 25x25 region
    for i in range(size):
        for j in range(size):
            if (half_center <= i < size - half_center) and (half_center <= j < size - half_center):
                weights[i, j] = center_weight_factor  # Apply high weight to the center region

    # Apply weights to the black pixels and sum
    weighted_black_pixels = np.sum((cell == 0) * weights)

    print(f"Weighted black pixels in cell: {weighted_black_pixels}")
    return weighted_black_pixels < black_pixel_threshold




def extract_empty_cells(grid_cells):
    """
    Identifies empty cells from a list of grid cells and returns their indices.
    
    Args:
        grid_cells (list): List of 50x50 binary images.
    
    Returns:
        set: Set of indices of empty grid cells.
    """
    empty_cells_indices = set()
    for idx, cell in enumerate(grid_cells):
        if is_empty_cell(cell):
            empty_cells_indices.add(idx)
            print(f"Cell {idx} is empty.")
        else:
            print(f"Cell {idx} contains digit.")
    return empty_cells_indices


def get_81_grid_cells_from_sudoku_grid(image_path):
    # Load and threshold the image
    binary_image = load_and_threshold_image(image_path)
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Remove large black lines
    # result = remove_large_black_lines(binary_image)
    # cv2.imshow("Result Image", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Apply adaptive filtering
    cleaned_result = adaptively_filter_image(binary_image)
    cv2.imshow("Cleaned Image", cleaned_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Extract grid cells
    grid_cells = extract_grid_cells(cleaned_result)
    empty_cells_indices = extract_empty_cells(grid_cells)
    # show non empty cells
    for idx, cell in enumerate(grid_cells):
        if idx in empty_cells_indices:
            continue
        cv2.imshow(f"Cell {idx}", cell)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"Extracted {len(grid_cells)} cells.")
    
get_81_grid_cells_from_sudoku_grid("test7.jpg")
# get_81_grid_cells_from_sudoku_grid("images.png")