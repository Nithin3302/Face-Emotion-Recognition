# 🎭 Face Emotion Recognition AI

A Streamlit-based web application that detects and classifies facial emotions in real-time using deep learning. The application supports both webcam feed and image upload functionality.

## 🔍 Features
- Real-time emotion detection from webcam
- Upload and analyze images
- Support for 7 different emotions:
  - 😠 Angry
  - 🤢 Disgust
  - 😨 Fear
  - 😊 Happy
  - 😢 Sad
  - 😲 Surprise
  - 😐 Neutral
- FPS counter for performance monitoring
- Confidence scores for predictions

## 🧠 Model
- TensorFlow/Keras CNN model
- Trained on FER2013 dataset
- Input size: 48x48 grayscale images
- Uses OpenCV for face detection and preprocessing

## 📊 Datasets
- FER2013 (Facial Expression Recognition Dataset)
- 7 emotion classes
- Grayscale images of faces
- Suitable for real-world applications

## 🖥️ Installation & Usage

### Prerequisites
This repository uses Git LFS for managing large files. Before cloning, install Git LFS:
```bash
git lfs install
```

### Setup
```bash
# Clone the repository
git clone https://github.com/Nithin3302/Face-Emotion-Recognition.git
cd Face-Emotion-Recognition

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/app.py
```

## 📂 Project Structure
```
face-emotion-recognition/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py  # Dataset loading utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── emotion_model.py # Model architecture
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py # Image preprocessing
├── saved_models/
│   └── emotion_model.h5    # Trained model weights
├── tests/
│   ├── __init__.py
│   └── test_model.py
└── requirements.txt
```

## 📦 Requirements
- 🎯 TensorFlow >= 2.9.0 - Deep learning framework
- 🖼️ OpenCV >= 4.6.0 - Image processing
- 🔢 NumPy >= 1.21.0 - Numerical computations
- 📊 Pandas >= 1.3.0 - Data handling
- 🧮 Scikit-learn >= 0.24.2 - Machine learning utilities
- 📈 Matplotlib >= 3.4.2 - Plotting
- 🖼️ Pillow >= 8.3.1 - Image processing
- 🌐 Streamlit - Web interface

## 👥 Contributors
- Nithin R Poojary
