import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from keras import backend as K  # Import K from keras.backend

from custom_datagenerator import imageLoader
from tensorflow import keras
from simple_3d_Unet import simple_unet_model
import segmentation_models_3D as sm  # Import segmentation_models_3D

# Directories and file lists
train_img_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128\train\images"
train_mask_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128\train\masks"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(img_list)

img_num = random.randint(0, num_images - 1)
test_img = np.load(os.path.join(train_img_dir, img_list[img_num]))
test_mask = np.load(os.path.join(train_mask_dir, msk_list[img_num]))
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2] - 1)
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

#####################################################
train_img_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128\train\images"
train_mask_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128\train\masks"

val_img_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128\val\images"
val_mask_dir = r"C:\Users\computer house\Downloads\Segmentation of BraTS2020 Project\BraTS2020_TrainingData\input_data_128\val\masks"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
##################################

########################################################################
batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list,
                                val_mask_dir, val_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2] - 1)
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

###########################################################################
# Define loss, metrics and optimizer to be used for training

wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))

# Define a custom categorical focal loss function with K.pow instead of ops.pow
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    """
    Softmax version of focal loss.
       m
      ----
      \         alpha * (1 - p_{i})^gamma * y_{i} * log(p_{i})
      /__
       i

    Where m is the number of classes,
          alpha is the focusing parameter,
          gamma is the modulation parameter and
          y is the ground truth probability distribution
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Clip the predictions to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

focal_loss = categorical_focal_loss()

# Use CombinedLoss to combine dice_loss and focal_loss
total_loss = sm.losses.CombinedLoss(losses=[dice_loss, focal_loss], loss_weights=[1, 1])

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)
#######################################################################
# Fit the model

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

model = simple_unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=128,
                          IMG_CHANNELS=3,
                          num_classes=4)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch)

model.save('brats_3d.hdf5')

# plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#################################################
from keras.models import load_model

# Load model for prediction or continue training

# For continuing training....
# The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
# This is because the model does not save loss function and metrics. So to compile and
# continue training we need to provide these as custom_objects.
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5')

# So let us add the loss as custom object... but the following throws another error...
# Unknown metric function: iou_score
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss})

# Now, let us add the iou_score function we used during our initial training
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score': sm.metrics.IOUScore(threshold=0.5)})

# Now all set to continue the training process.
history2 = my_model.fit(train_img_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=1,
                        verbose=1,
                        validation_data=val_img_datagen,
                        validation_steps=val_steps_per_epoch)

#################################################
# For predictions you do not need to compile the model, so ...
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5',
                      compile=False)

# Verify IoU on a batch of images from the test dataset
# Using built-in keras function for IoU
# Only works on TF > 2.0
from keras.metrics import MeanIoU

batch_size = 8  # Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list,
                               val_mask_dir, val_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#############################################
# Predict on a few test images, one at a time
# Try images:
img_num = 82

test_img = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_" + str(img_num) + ".npy")

test_mask = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_" + str(img_num) + ".npy")
test_mask_argmax = np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]


# Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

# n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:, :, n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:, :, n_slice])
plt.show()
