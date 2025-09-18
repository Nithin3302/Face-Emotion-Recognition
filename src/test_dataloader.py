from data.data_loader import FERDataLoader

# Initialize data loader
data_loader = FERDataLoader('datasets')

# Load the data
X_train, X_test, y_train, y_test = data_loader.load_data()

# Print shapes to verify
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")