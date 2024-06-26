import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Define the path to your training data
TRAIN_DATASET_PATH = 'C:/Users/computer house/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

# Load the FLAIR image
# test_image_flair = TRAIN_DATASET_PATH + 'BraTS20_Training_355/BraTS20_Training_355_flair.nii'
test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_355/BraTS20_Training_355_flair.nii').get_fdata()
print(test_image_flair.max())

plt.imshow(test_image_flair[:, :, test_image_flair.shape[2] // 2], cmap='gray')
plt.title("Middle Slice of FLAIR Image")
plt.show()
# try:
#     test_image_flair = nib.load(flair_path).get_fdata()
#     print("Data loaded successfully.")
#     print(f"Data shape: {test_image_flair.shape}")
#     print(f"Max value in data: {test_image_flair.max()}")
#
#     # Optionally, you can visualize a slice of the image to confirm it looks correct
#     plt.imshow(test_image_flair[:, :, test_image_flair.shape[2] // 2], cmap='gray')
#     plt.title("Middle Slice of FLAIR Image")
#     plt.show()
# except FileNotFoundError:
#     print(f"File not found: {flair_path}")
# except Exception as e:
#     print(f"An error occurred: {e}")
#
# TRAIN_DATASET_PATH = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
# #VALIDATION_DATASET_PATH = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
#
# test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_355/BraTS20_Training_355_flair.nii').get_fdata()
#
# print(test_image_flair.max())
