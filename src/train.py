import os
import matplotlib.pyplot as plt
from data.data_loader import FERDataLoader
from models.emotion_model import EmotionCNN

def plot_training_history(history):
    """Plot training & validation accuracy and loss values"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')

def main():
    # Create save directory if it doesn't exist
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    # Initialize data loader
    print("Loading dataset...")
    data_loader = FERDataLoader('datasets')
    X_train, X_test, y_train, y_test = data_loader.load_data()
    
    # Build the model
    print("Building model...")
    model = EmotionCNN.build_model(
        width=48,
        height=48,
        depth=1,
        classes=7
    )
    
    # Display model summary
    model.summary()
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Save the model
    model.save('saved_models/emotion_model.h5')
    print("Model saved to saved_models/emotion_model.h5")
    
    # Plot training history
    plot_training_history(history)
    print("Training history plot saved as training_history.png")

if __name__ == "__main__":
    main()