# Face Emotion Recognition AI 🎭

A real-time facial emotion detection system built with TensorFlow, OpenCV, and Streamlit. The application can detect emotions from both live webcam feed and uploaded images.

## Features ✨

- Real-time emotion detection from webcam
- Image upload and emotion detection
- Support for 7 different emotions
- Clean and modern UI with dark theme
- FPS counter for performance monitoring
- Confidence scores for predictions

## Supported Emotions 😊

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

## Tech Stack 🛠️

- Python 3.10+
- TensorFlow 2.9.0
- OpenCV
- Streamlit
- Numpy
- Pandas
- Scikit-learn
- FER2013 Dataset

## Installation 📦

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-emotion-recognition.git
cd face-emotion-recognition
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Start the Streamlit app:
```bash
streamlit run src/app.py
```

2. Use the application:
   - Click "Start Camera" for real-time emotion detection
   - Or upload an image using the "Choose an image..." option
   - View emotion predictions and confidence scores

## Project Structure 📁

```
face-emotion-recognition/
├── README.md
├── requirements.txt
├── datasets/
│   ├── train/
│   └── test/
├── saved_models/
│   └── emotion_model.h5
└── src/
    ├── app.py
    ├── data/
    │   ├── __init__.py
    │   └── data_loader.py
    ├── models/
    │   ├── __init__.py
    │   └── emotion_model.py
    └── utils/
        ├── __init__.py
        └── preprocessing.py
```

## Model Information 🧠

The emotion detection model is trained on the FER2013 dataset and uses a Convolutional Neural Network (CNN) architecture. The model achieves good accuracy in detecting seven different emotional states from facial expressions.

## Performance 📊

- Real-time processing at 15-30 FPS (depending on hardware)
- Emotion detection confidence scores
- Support for multiple face detection

## Requirements 📋

- Python 3.10 or higher
- Webcam (for real-time detection)
- Modern web browser
- Minimum 4GB RAM
- CPU with SSE4.2 support

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- FER2013 dataset providers
- TensorFlow and OpenCV communities
- Streamlit framework developers

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact 📧

Your Name - your.email@example.com
Project Link: [https://github.com/yourusername/face-emotion-recognition](https://github.com/yourusername/face-emotion-recognition)