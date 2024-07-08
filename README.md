# Emotion Recognition from Webcam

## Overview

This project is a computer vision application that captures video from your webcam, detects faces, and classifies the emotions displayed on those faces. The project uses deep learning models for face detection and emotion classification, packaged in a Docker container for easy setup and deployment.

## Features

- Real-time face detection using OpenCV
- Emotion classification using a pre-trained deep learning model
- Easy-to-use Docker setup for consistent environment
- Modular and extensible codebase

## Project Structure

```
emotion-recognition-app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── camera.py
│   ├── face_detection.py
│   ├── emotion_classification.py
│   └── utils.py
├── models/
│   ├── lightning_emotion_vgg.py
│   ├── emotion_classification_model.pth
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.8 or later
- Docker

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/emotion-recognition-app.git
   cd emotion-recognition-app
   ```

2. **Build and Run with Docker**

   Make sure Docker is installed and running on your system.

   ```bash
   docker-compose up --build
   ```

3. **Run Locally without Docker**

   If you prefer to run the project without Docker, follow these steps:

   - **Create a virtual environment and activate it:**

     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```

   - **Install dependencies:**

     ```bash
     pip install -r requirements.txt
     ```

   - **Run the application:**

     ```bash
     python app/main.py
     ```

## Usage

Once the application is running, it will open a window displaying the webcam feed. Detected faces will be marked with rectangles, and the classified emotion will be displayed above each face.

To stop the application, close the webcam window or press `q`.

## Models

- **Face Detection**: Uses OpenCV's Haar Cascade Classifier for real-time face detection.
- **Emotion Classification**: Uses a custom-trained deep learning model (`lightning_emotion_vgg.py`) to classify emotions. The model is pre-trained and saved as `emotion_classification_model.pth`.

## Training Your Own Models

If you want to train your own models, you can use the provided Jupyter notebooks in the `notebooks/` directory:

- `data_preprocessing.ipynb`: Preprocess the dataset for training.
- `model_training.ipynb`: Train the emotion classification model.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.
