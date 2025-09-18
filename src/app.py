import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time

# Set page configuration and theme
st.set_page_config(
    page_title="Emotion Detection AI",
    layout="wide",
)

# Custom CSS for red and black theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #cc0000;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #ff0000;
    }
    .upload-text {
        color: #cc0000;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stats-container {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #ffffff;
    }
    .centered-header {
        text-align: center;
        color: #ffffff;
        padding: 1rem 0;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
    }
    .main-title {
        text-align: center;
        color: #cc0000;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

class EmotionDetector:
    def __init__(self):
        try:
            self.model = load_model('saved_models/emotion_model.h5')
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise

    def detect_emotion(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        emotions_detected = []
        confidence_scores = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = roi_gray.astype('float32') / 255.0
            
            # Predict emotion
            prediction = self.model.predict(roi_gray)
            emotion_idx = np.argmax(prediction)
            emotion = self.emotions[emotion_idx]
            confidence = float(prediction[0][emotion_idx])
            
            emotions_detected.append(emotion)
            confidence_scores.append(confidence)
            
            # Draw rectangle and emotion text
            cv2.rectangle(image, (x, y), (x+w, y+h), (204, 0, 0), 2)
            text = f"{emotion} ({confidence:.2%})"
            cv2.putText(image, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 0), 2)
        
        return image, emotions_detected, confidence_scores

def main():
    # Title with custom styling
    st.markdown('<h1 class="main-title">🎭 Emotion Detection AI</h1>', unsafe_allow_html=True)
    
    detector = EmotionDetector()
    
    # Live Camera Section
    st.markdown('<h2 class="centered-header">📸 Live Camera</h2>', unsafe_allow_html=True)
    
    with st.container():
        # Create columns for buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Initialize session state for camera
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
        
        # Show Start/Stop button based on camera state
        with col2:
            if not st.session_state.camera_on:
                if st.button('Start Camera', key='start_btn'):
                    st.session_state.camera_on = True
                    st.rerun()  # Changed from experimental_rerun
            else:
                if st.button('Stop Camera', key='stop_btn'):
                    st.session_state.camera_on = False
                    st.rerun()  # Changed from experimental_rerun
        
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        if st.session_state.camera_on:
            try:
                cap = cv2.VideoCapture(0)
                
                while cap.isOpened() and st.session_state.camera_on:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            processed_frame, emotions, confidences = detector.detect_emotion(frame)
                            
                            video_placeholder.image(
                                processed_frame,
                                channels='RGB',
                                width='stretch'
                            )
                            
                            if emotions and len(emotions) > 0:
                                stats_placeholder.markdown(
                                    f"""
                                    <div class="stats-container">
                                        <p>Detected Emotion: {emotions[-1]}</p>
                                        <p>Confidence: {confidences[-1]:.2%}</p>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                        except Exception as e:
                            st.error(f"Error processing frame: {str(e)}")
                            break
                            
                        time.sleep(0.01)
            
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                st.session_state.camera_on = False
            
            finally:
                try:
                    cap.release()
                except:
                    pass
                
                try:
                    video_placeholder.empty()
                    stats_placeholder.empty()
                except:
                    pass
                
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except:
                    pass

    # Add a separator between sections
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Image Upload Section
    st.markdown('<h2 class="centered-header">🖼️ Image Upload</h2>', unsafe_allow_html=True)
    
    # Center the upload button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        processed_image, emotions, confidences = detector.detect_emotion(image)
        
        # Center the processed image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(processed_image, caption='Processed Image', channels='RGB')
            
            # Display detection results
            if emotions:
                st.markdown("""
                    <div class="stats-container">
                        <h3 style="color: #ffffff; text-align: center;">Detection Results</h3>
                """, unsafe_allow_html=True)
                for emotion, confidence in zip(emotions, confidences):
                    st.markdown(f"<p style='text-align: center; color: #ffffff;'>- {emotion}: {confidence:.2%}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # About section at the bottom
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("ℹ️ About"):
        st.markdown("""
        <div style='color: #ffffff; font-family: Roboto, sans-serif;'>
        <h3>How it works</h3>
        <ul>
            <li>Uses deep learning to detect facial expressions</li>
            <li>Supports 7 different emotions</li>
            <li>Real-time processing capabilities</li>
            <li>Built with TensorFlow and OpenCV</li>
        </ul>
        
        <h3>Supported Emotions</h3>
        <ul>
            <li>😠 Angry</li>
            <li>🤢 Disgust</li>
            <li>😨 Fear</li>
            <li>😊 Happy</li>
            <li>😢 Sad</li>
            <li>😲 Surprise</li>
            <li>😐 Neutral</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()