from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

class EmotionCNN:
    @staticmethod
    def build_model(width, height, depth, classes):
        model = Sequential([
            # First Convolution Block
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(width, height, depth)),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Second Convolution Block
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Flatten and Dense Layers
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(classes, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model