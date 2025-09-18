import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class FERDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.num_classes = 7
        self.emotion_map = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
        }

    def load_data(self):
        """
        Load and preprocess images from directory structure
        Returns:
            X_train, X_test: Training and test images
            y_train, y_test: Training and test labels
        """
        images = []
        labels = []

        # Load training data
        train_path = os.path.join(self.data_path, 'train')
        print("Loading training data...")

        # Get total number of images for progress bar
        total_images = sum([len(files) for r, d, files in os.walk(train_path)])

        with tqdm(total=total_images) as pbar:
            for emotion_folder in os.listdir(train_path):
                folder_path = os.path.join(train_path, emotion_folder)
                if os.path.isdir(folder_path):
                    for image_file in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, image_file)
                        try:
                            # Load and preprocess image
                            img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
                            img_array = img_to_array(img)
                            images.append(img_array)
                            labels.append(self.emotion_map[emotion_folder])
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error loading {image_path}: {str(e)}")

        # Convert to numpy arrays
        print("\nConverting to numpy arrays...")
        X = np.array(images, dtype='float32') / 255.0
        y = to_categorical(np.array(labels), self.num_classes)

        # Split into train and test sets
        print("Splitting into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test