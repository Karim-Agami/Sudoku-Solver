import pickle
import cv2
import numpy as np
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

def split_image_to_grid(image, grid_size=(9, 9)):
    """
    Splits an image into a grid of equal parts.

    Parameters:
    image (numpy.ndarray): A numpy array of shape (H, W) representing the input image.

    Returns:
    List[numpy.ndarray]: A list of numpy arrays of shape (h, w) representing the grid of images.
    """
    h, w = image.shape
    rows, cols = grid_size
    h_step = h // rows
    w_step = w // cols

    grid = []
    for i in range(0, h, h_step):
        for j in range(0, w, w_step):
            grid.append(image[i:i+h_step, j:j+w_step])

    return np.array(grid)

def predict(model_name, image):
    """
    Predicts the class of the input images using the model.

    Parameters:
    model_name (str): The name of the file containing the model.
    images (List[numpy.ndarray]): A list of numpy arrays of shape (H, W) representing the input images.

    Returns:
    numpy.ndarray: A 2D numpy array where each element is the predicted class of the corresponding image in the input list, reshaped according to the grid size.
    
    """
    model=load_model(model_name)
    images=split_image_to_grid(image)
    features=extract_features_for_all(images)
    result=prediction_from_grid(features, model)
    result=reshape_predictions(result)
    return result
