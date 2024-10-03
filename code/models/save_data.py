import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define the path to save the dataset
datasets_folder = os.path.abspath(os.path.join(os.getcwd(), "code/datasets"))

# Ensure the 'datasets' folder exists
os.makedirs(datasets_folder, exist_ok=True)

# Save the dataset to a .npz file
np.savez_compressed(os.path.join(datasets_folder, 'cifar10_data.npz'),
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)