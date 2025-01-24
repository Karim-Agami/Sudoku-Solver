# Table of Contents
1. [Overview](#overview) 🌟  
2. [Features](#features) ✨  
3. [How It Works](#how-it-works) 🛠️  
4. [Technologies Used](#technologies-used) 🖥️  
5. [Why Use This App?](#why-use-this-app) ❓  
6. [Sample Results](#sample-results) 📸  

---

## Overview 🌟  
This application makes solving Sudoku puzzles easy and fun! Just snap a photo or upload one from your gallery, and our app will analyze the puzzle and provide you with the solution. 🎉  

---

## Features ✨  
- **Photo Input**: Capture a Sudoku puzzle directly using your camera. 📸  
- **Gallery Upload**: Select an existing image of a puzzle from your device's gallery. 🖼️  
- **Automatic Solution**: The app detects the grid, solves the puzzle, and displays the solution. 🤖  
- **Simple Interface**: Designed to be user-friendly for everyone. 👩‍💻👨‍💻  

---

## How It Works 🛠️  
1. **Upload a Photo**: Choose to capture a new image or select one from your gallery.  
2. **Image Processing**: The app identifies the Sudoku grid and prepares it for digit extraction.  
3. **Digit Extraction with Machine Learning**: Using techniques like HOG for feature extraction and a trained classifier, the app recognizes and extracts the digits from the grid.  
4. **Sudoku Solver**: Using a smart algorithm, the app solves the puzzle.  
5. **Solution Display**: The solved grid is shown on your screen using a Flutter application.  

---

## Technologies Used 🖥️  
- **Image Processing**:  
  - **OpenCV**: For grid detection, preprocessing, and digit extraction.  
  - **Scikit-image (skimage)**: For advanced image processing tasks like HOG (Histogram of Oriented Gradients) feature extraction.  
- **Machine Learning**:  
  - **Scikit-learn**: For training and deploying machine learning models (e.g., SVM) for digit recognition.  
  - **PyTorch**: For leveraging pre-trained models from Hugging Face for enhanced accuracy.  
- **Application Development**:  
  - **Flutter**: For building a cross-platform mobile application with a seamless user interface.  
- **Programming Language**:  
  - **Python**: For backend image processing, machine learning, and Sudoku-solving logic.  

---

## Sample Results 📸  

Here are some examples of the Sudoku solver app in action, showcasing the original puzzle and the solved result.  

### Example from camera  
<img src="https://github.com/user-attachments/assets/b475f124-a32a-4d01-a59c-3f8a29f2a9cd" alt="WhatsApp Image 2024-12-23 at 22 22 11_e3e0e796" width="300" />

### Example with rotated image from camera  
<img src="https://github.com/user-attachments/assets/48435c22-c208-44bc-9562-4a32cee009b0" alt="WhatsApp Image 2024-12-23 at 22 24 45_c1b53851" width="300" />

### Example from gallery  
<img src="https://github.com/user-attachments/assets/ecd454de-fd23-4b2e-b419-9797a8a66b07" alt="WhatsApp Image 2024-12-23 at 21 28 16_3dbdb8f7" width="300" />

---

## Why Use This App? ❓  
- Save time by solving puzzles automatically. ⏳  
- Get accurate solutions in seconds. 🎯  
- Enjoy a smooth and intuitive experience. 😊  

---

Try it out today and let the app do the hard work for you! 🚀  
