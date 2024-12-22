import time
import cv2
import numpy as np
from CandidateMinimizationSolver import solve_sudoku_int
from model import predict 
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

# def load_and_threshold_image(image_path):
  
#     # Step 1: Load the image as RGB and resize to 450x450
#     rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if rgb_image is None:
#         raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
#     rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     rgb_image = cv2.resize(rgb_image, (450, 450))           # Resize to 450x450 pixels
    
#     # Step 2: Convert the image to grayscale
#     grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
#     # Step 3: Apply adaptive thresholding
#     binary_image = cv2.adaptiveThreshold(
#         grayscale_image,           # Source image
#         255,                       # Maximum value to assign
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method (Gaussian)
#         cv2.THRESH_BINARY,         # Threshold type (binary thresholding)
#         11,                        # Block size (size of neighborhood)
#         2                          # Constant to subtract from mean
#     )
    
#     return binary_image

#note if you change this pls also change the is_empty pixel count
def extract_grid_cells(result_image, cell_size=50, crop_size=40):
    
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
   
    # Smooth the image with a Gaussian filter to remove noise
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Compute the difference between the original and smoothed images
    noise_image = cv2.absdiff(image, smoothed_image)
    
    # Calculate the standard deviation of the noise
    noise_level = np.std(noise_image)
    
    return noise_level

def adaptively_filter_image(image):
    
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


def is_empty_cell(cell, black_pixel_threshold=1000, center_weight_factor=100):

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
 
    empty_cells_indices = set()
    for idx, cell in enumerate(grid_cells):
        if is_empty_cell(cell):
            empty_cells_indices.add(idx)
            print(f"Cell {idx} is empty.")
        else:
            print(f"Cell {idx} contains digit.")
    return empty_cells_indices


# def get_empty_cells_indices(image_path):
#     # Load and threshold the image
#     binary_image = load_and_threshold_image(image_path)
#     cv2.imshow("Binary Image", binary_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # Remove large black lines
#     # result = remove_large_black_lines(binary_image)
#     # cv2.imshow("Result Image", result)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
    
#     # Apply adaptive filtering
#     cleaned_result = adaptively_filter_image(binary_image)
#     cv2.imshow("Cleaned Image", cleaned_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Extract grid cells
#     grid_cells = extract_grid_cells(cleaned_result)
#     empty_cells_indices = extract_empty_cells(grid_cells)
#     # show non empty cells
#     for idx, cell in enumerate(grid_cells):
#         if idx in empty_cells_indices:
#             continue
#         cv2.imshow(f"Cell {idx}", cell)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     print(f"Extracted {len(grid_cells)} cells.")
#     return empty_cells_indices

# def get_non_empty_grid_cells_as_grayscale(image_path):
#     image=cv2.imread(image_path)
#     cleaned_result = adaptively_filter_image(image)
    
#     # Extract grid cells
#     grid_cells = extract_grid_cells(cleaned_result)
#     empty_cells_indices = extract_empty_cells(grid_cells)
    
#     # Extract non-empty cells as grayscale images
#     non_empty_cells = []
#     for idx, cell in enumerate(grid_cells):
#         if idx not in empty_cells_indices:
#             non_empty_cells.append(cell)
#     return non_empty_cells

def get_non_empty_grid_cells_as_grayscale(image_path):
    # Step 1: Load the original image in grayscale
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    resized_image = cv2.resize(grayscale_image, (450, 450))

    # Step 2: Perform adaptive filtering for noise reduction
    cleaned_result = adaptively_filter_image(resized_image)

    # Step 3: Create a binary image for empty cell detection
    binary_image = cv2.adaptiveThreshold(
        cleaned_result, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    # Step 4: Extract grid cells from the binary image
    binary_cells = extract_grid_cells(binary_image)
    
    # Step 5: Identify empty cells using the binary cells
    empty_cells_indices = extract_empty_cells(binary_cells)

    # Step 6: Use the grayscale image for non-empty cells
    grayscale_cells = extract_grid_cells(cleaned_result)
    
    # Step 7: Extract non-empty cells as grayscale images
    non_empty_cells = [
        grayscale_cells[idx] 
        for idx in range(len(grayscale_cells)) 
        if idx not in empty_cells_indices
    ]
    
    # # Step 8: Display the non-empty cells
    # for idx, cell in enumerate(non_empty_cells):
    #     cv2.imshow(f"Cell {idx}", cell)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    print(f"Total cells: {len(grayscale_cells)}, Non-empty cells: {len(non_empty_cells)}")
    return non_empty_cells, empty_cells_indices

def construct_sudoku_grid(numbers_from_cells, empty_cells_indices):
    # Create a 9x9 grid with zeros
    grid = np.zeros((9, 9), dtype=int)
    
    # Keep track of the index in the numbers_from_cells list
    numbers_index = 0
    
    # Iterate through all indices of a 9x9 grid
    for idx in range(81):  # Total cells in a 9x9 grid
        if idx in empty_cells_indices:
            # Skip this cell as it's empty
            continue
        # Otherwise, populate the grid with a number from numbers_from_cells
        row = idx // 9
        col = idx % 9
        grid[row][col] = numbers_from_cells[numbers_index]
        numbers_index += 1
    
    return grid

def resize_non_empty_cells(non_empty_cells):
    resized_cells = []
    for cell in non_empty_cells:
        resized_cell = cv2.resize(cell, (32, 32))
        resized_cells.append(resized_cell)
    return resized_cells

def print_sudoku_grid(grid):
    """
    Print a 9x9 Sudoku grid in a clean and visually appealing format.
    
    Args:
        grid (list of list of int): A 9x9 matrix representing the Sudoku grid.
    """
    print("+-------+-------+-------+")
    for i in range(9):
        row = "| "
        for j in range(9):
            cell = grid[i][j]
            row += f"{cell if cell != 0 else '.'} "  # Use '.' for empty cells
            if (j + 1) % 3 == 0:
                row += "| "
        print(row)
        if (i + 1) % 3 == 0:
            print("+-------+-------+-------+")

def get_solved_sudoku_from_image(image_path):
    non_empty_cells, empty_cells_indices = get_non_empty_grid_cells_as_grayscale(image_path)
    # Resize the non-empty cells to 32x32
    non_empty_cells_rezied = resize_non_empty_cells(non_empty_cells)
    #call ocr model with the non empty images
    numbers_from_cells =predict("knn_model.pkl",non_empty_cells_rezied)
    #mock the number form cells for now
    # numbers_from_cells=[8, 6, 9, 5, 9, 6, 2, 7, 1, 2, 5, 7, 4, 3, 3, 9, 2, 3, 1, 5, 8, 6, 9, 2, 5, 3]
    grid=construct_sudoku_grid(numbers_from_cells, empty_cells_indices)
    print("Sudoku grid constructed successfully.")
    print_sudoku_grid(grid)
    print("-------------------------------------------------------------------------------------")
    # Solve the Sudoku grid
    solve_status = solve_sudoku_int(grid)
    if solve_status is 1:
        print("Sudoku grid solved successfully.")
        print_sudoku_grid(grid)
        return grid
    elif solve_status is -2:
        print("Sudoku grid is not solvable.")
        return solve_status
    elif solve_status is -1:
        print("Invalid Sudoku grid.")
        return solve_status
    else:
        print("Should Not ever Reach This Statement")
    
if __name__ == "__main__":
    #measure time
    time_now = time.time()
    image_path = "download.png"
    get_solved_sudoku_from_image(image_path)
    print(f"Time taken: {time.time() - time_now:.2f} seconds.")