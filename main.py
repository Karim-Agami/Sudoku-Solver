import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import uvicorn
import cv2
import numpy as np
app = FastAPI()
import matplotlib.pyplot as plt
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
def order_points(pts):
    pts = pts.reshape((4, 2))
    # Sort the points based on the sum and difference of the coordinates
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of the points to find the top-left and bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # Difference of the points to find the top-right and bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def detect_lines_in_contour(image):

    # Detect edges using Canny edge detector
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    return lines
def line_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle

def count_perpendicular_lines(lines):
    if lines is None:
        return 0, 0, None

    vertical_lines = []
    horizontal_lines = []
    other_lines = []

    # Separate lines into vertical, horizontal, and other lines based on angle
    for line in lines:
        angle = line_angle(line)
        if 80 < abs(angle) < 100:  # Vertical-like lines
            vertical_lines.append(line)
        elif abs(angle) < 10 or abs(angle) > 170:  # Horizontal-like lines
            horizontal_lines.append(line)
        else:
            other_lines.append(line)

    # Check intersections between perpendicular lines
    intersections = []
    
    # Define a helper function to calculate the slope of a line
    def calculate_slope(line):
        x1, y1, x2, y2 = line[0]
        if x2 == x1:  # Vertical line
            return None
        elif y2 == y1:  # Horizontal line
            return 0
        else:
            return (y2 - y1) / (x2 - x1)

    # Check perpendicularity for vertical and horizontal lines
    for v_line in vertical_lines:
        for h_line in horizontal_lines:
            x1_v, y1_v, x2_v, y2_v = v_line[0]
            x1_h, y1_h, x2_h, y2_h = h_line[0]
            
            # Calculate the intersection points
            intersection_x = x1_v
            intersection_y = y1_h
            intersections.append((intersection_x, intersection_y))
    
    # Check perpendicularity for other lines with each other (using slope check)
    for line1 in other_lines:
        slope1 = calculate_slope(line1)
        for line2 in other_lines:
            if np.array_equal(line1, line2):

                continue  # Skip comparing the same line with itself
            slope2 = calculate_slope(line2)
            if slope1 is not None and slope2 is not None and slope1 * slope2 == -1:
                # Lines are perpendicular based on slopes
                intersections.append((line1, line2))  # You can store the actual lines if needed

    return len(vertical_lines), len(horizontal_lines), len(other_lines), intersections



def grid_detect(image):

    # show_images([image], ["Original Image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show_images([gray], ["Gray Image"])

    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    # show_images([blur], ["Blur Image"])

    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    # show_images([thresh], ["Threshold Image"])
    
    # eroded= cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    # # dialated= cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=1)
    # show_images([eroded], ["Dilated Image"])
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
     # Set your desired minimum area for the contour
    valid_contours = []
    image_area = image.shape[0] * image.shape[1]
    min_area = image_area//15
    print(len(contours))
    # Filter out contours based on area
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_area and area < image_area :  # Exclude contours near image size
            valid_contours.append(contour)

    max_area = 0
    c = 0
    best_cnt = None
    warped_final = None
    output_image = image.copy()
    for i,contour in enumerate(valid_contours):
        cv2.drawContours(output_image, [contour], -1, (255, 0, 0), 2) 
        # show_images([output_image], ["Valid Contours"])
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        print(approx.shape[0])
        if len(approx) == 4:  # Check if the contour is quadrilateral
            print("Quadrilateral found")
            image_with_lines = image.copy()
            approx = order_points(approx)
            # if not is_square(approx):
            #     print("Not a square")
            #     continue

            pts1 = np.float32(approx)  # Contour points
            pts2 = np.float32([[0, 0], [288, 0], [288, 288], [0, 288]])  # Destination points for top-down view

            # Apply the perspective transform
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(image, matrix, (288, 288))  # Use the original image here

            # Show the warped image for each detected contour
            # show_images([warped], ["Warped Image"])

            # Now, check for the sudoku grid in the warped image (9x9 grid)
            # This can be done by detecting lines or dividing the image into 9x9 cells
            lines = detect_lines_in_contour(warped)
            if lines is not None and len(lines) > 0:
                v_count, h_count, _,_ = count_perpendicular_lines(lines)
                # print(f"Detected {v_count} vertical lines and {h_count} horizontal lines.")
        
        # Check if there are at least 9 vertical lines and 9 horizontal lines
            
                if v_count >= 9 and h_count >= 9:
                    print("9x9 grid detected!")
                    current_area = cv2.contourArea(contour)
                    if best_cnt is None or current_area > cv2.contourArea(best_cnt):
                            print(f"New best contour found with area {current_area}.")
                            best_cnt = contour  # Update the best contour
                            warped_final = warped
                else:
                    print("9x9 grid not detected.")
    else:
        print("No lines detected.")

        c += 1

    # After processing all contours, use best_cnt for further operations
    if best_cnt is not None and warped_final is not None:
        mask = np.zeros((warped_final.shape[0], warped_final.shape[1]), np.uint8)  # Use warped image dimensions
        cv2.drawContours(mask, [best_cnt], 0, 255, -1)
        cv2.drawContours(mask, [best_cnt], 0, 0, 2)

        out = np.zeros_like(warped_final)
        out[mask == 255] = warped_final[mask == 255]

        approx = cv2.approxPolyDP(best_cnt, 0.02 * cv2.arcLength(best_cnt, True), True)

        image2 = warped_final.copy()

        # show_images([image2], ["Debug - Approx Points"])
        return image2
        # image_to_save = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Pillow

        # # Create a PIL Image object
        # pil_image = Image.fromarray(image_to_save)

        # # Save the image as a JPG file
        # output_path = "debug_approx_points.jpg"
        # pil_image.save(output_path)

        # print(f"Image saved as {output_path}")

        # Ensure we have a quadrilateral in the warped image
        if len(approx) == 4:
            approx = order_points(approx)
            pts1 = np.float32(approx)  # Contour points
            pts2 = np.float32([[0, 0], [288, 0], [288, 288], [0, 288]])  # Destination points for top-down view

            # Apply the perspective transform again on the warped image
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped_final = cv2.warpPerspective(warped_final, matrix, (288, 288))
            show_images([image2], ["Warped Final Image"])
        else:
            print("Grid not detected correctly!")
    else:
        print("No valid contours found.")


def divide_image(image):

# Assuming `opencv_image` is your loaded image
    height, width = image.shape[:2]  # Get dimensions of the image

    # Calculate the dimensions of each cell
    cell_width = width // 9
    cell_height = height // 9

    # List to store cells row by row
    cells = []

    # Loop to extract each 9x9 cell
    for row in range(9):
        row_cells = []  # List for the current row
        for col in range(9):
            # Calculate the boundaries of the current cell
            start_x = col * cell_width
            end_x = (col + 1) * cell_width
            start_y = row * cell_height
            end_y = (row + 1) * cell_height
            
            # Crop the cell from the image
            cell = image[start_y:end_y, start_x:end_x]
            
            # Add cell to the current row
            row_cells.append(cell)
        
        # Add the current row to the main list
        cells.append(row_cells)
    return cells
# Now `cells` contains a 9x9 grid of images stored row by row
import pickle
IMG_SIZE=32

def load_model(model_name):
    """
    Loads a model from disk.
    
    Parameters:
    model_name (str): The name of the file containing the model.
    
    Returns:
    object: The loaded model.
    """
    with open(model_name, 'rb') as f:
        return pickle.load(f)

def extract_hog_features(img):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from an image.

    Parameters:
    img (numpy.ndarray): A numpy array of shape (H, W) representing the input image.

    Returns:
    numpy.ndarray: A numpy array of shape (N,) containing the HOG features.
    """
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    win_size = (IMG_SIZE, IMG_SIZE)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)
    
    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9 
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()

def extract_features_for_all(images):
    """
    Extracts HOG features from a list of images.

    Parameters:
    images (List[numpy.ndarray]): A list of numpy arrays of shape (H, W) representing the input images.

    Returns:
    numpy.ndarray: A numpy array of shape (N, M) containing the HOG features for each image, where N is the number of images and M is the number of features per image.
    """
    features = []
    for img in images:
        features.append(extract_hog_features(img))
    return np.array(features)
    


def prediction_from_grid(features, model):
    result=model.predict(features)
    return result

def reshape_predictions(predictions, grid_size=(9, 9)):
    """
    Reshapes the predictions to a 2D grid.
    
    Parameters:
    predictions (numpy.ndarray): A 1D numpy array containing the predicted class of each image.
    grid_size (Tuple[int, int]): The size of the grid (rows, cols).
    
    Returns:
    numpy.ndarray: A 2D numpy array where each element is the predicted class of the corresponding image in the input list.
    """

    rows, cols = grid_size
    reshaped = np.array(predictions).reshape(rows, cols)
    return reshaped

import numpy as np

def split_image_to_grid(image, grid_size=(9, 9)):
    """
    Splits an image into a grid of equal parts.

    Parameters:
    image (numpy.ndarray): A numpy array of shape (H, W) representing the input image.
    grid_size (tuple): A tuple (rows, cols) specifying the number of grid cells along each dimension.

    Returns:
    List[numpy.ndarray]: A list of numpy arrays, each representing a grid cell.
    """
    h, w = image.shape
    rows, cols = grid_size
    h_step = h // rows
    w_step = w // cols

    grid = []
    for i in range(rows):
        for j in range(cols):
            # Handle edges by slicing up to the remaining dimensions for uneven division
            grid.append(image[i * h_step : (i + 1) * h_step if i < rows - 1 else h, 
                               j * w_step : (j + 1) * w_step if j < cols - 1 else w])

    return grid


def predict(model_name, image):
    """
    Predicts the class of the input images using the model.

    Parameters:
    model_name (str): The name of the file containing the model.
    images (List[numpy.ndarray]): A list of numpy arrays of shape (H, W) representing the input images.

    Returns:
    numpy.ndarray: A 2D numpy array where each element is the predicted class of the corresponding image in the input list, reshaped according to the grid size.
    
    """
    print("Predicting...")
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=cv2.resize(image, (288, 288))
    images=split_image_to_grid(image)
    try:
        model=load_model(model_name)
        print(model)
    except:
        print("Error loading model")
    

    
    
    features=extract_features_for_all(images)
    result=prediction_from_grid(features, model)
    result=reshape_predictions(result)
    return result

def process_image(image: Image.Image):
    # Convert the PIL image to an OpenCV-compatible format
    opencv_image = np.array(image)
    print("Input image shape:", opencv_image.shape)
    
    # Process the OpenCV image
    detected_grid = grid_detect(opencv_image)  # Call your existing image processing logic
    print("Detected grid shape:", detected_grid.shape)
    # Load the KNN model
    predictions = predict("knn_model.pkl", detected_grid)
    print(predictions)
    prediction_str = ",".join(predictions.flatten().astype(str))  # Convert to string and join
    prediction_str = prediction_str.replace(" ", "")  # Remove any spaces (if any)
    print(f"Formatted prediction string: {prediction_str}")
    return prediction_str
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read the file into a PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Process the image with your model
        result = process_image(image)
        return result  # Return the result as a string
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
