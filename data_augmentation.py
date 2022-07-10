
import os
import numpy as np

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset information

image_folder = os.path.join('datasets', 'face_dataset_train_images')
img_height, img_width = 250, 250  # size of images
batch_size = 16
save_format = 'jpg'

dataset = keras.preprocessing.image_dataset_from_directory(
    image_folder,
    seed=42,
    image_size=(img_height, img_width),
    label_mode='categorical',
    shuffle=True)

# Initial dataset is 195 images, but it can became several times larger

class_names = dataset.class_names
print(class_names)

n = 10

aug_image_folder = os.path.join('datasets', 'face_dataset_train_aug_images')
if not os.path.exists(aug_image_folder):
    os.makedirs(aug_image_folder)  # create folder if doesn't exist



def origin_image_to_aug_image(image_folder_to_generate, image_folder_to_save, n=10, batch_size=16,save_format='jpg'):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.7, 1),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')
    if not os.path.exists(image_folder_to_save):
        os.makedirs(image_folder_to_save)  # create folder if doesn't exist

    i = 0
    total = len(os.listdir(image_folder_to_generate))  # number of files in folder
    for filename in os.listdir(image_folder_to_generate):
        print("Step {} of {}".format(i+1, total))
        # for each image in folder: read it
        image_path = os.path.join(image_folder_to_generate, filename)
        image = keras.preprocessing.image.load_img(
            image_path, target_size=(img_height, img_width, 3))
        image = keras.preprocessing.image.img_to_array(
            image)  # from image to array
        # shape from (250, 250, 3) to (1, 250, 250, 3)
        image = np.expand_dims(image, axis=0)

        # create ImageDataGenerator object for it
        current_image_gen = train_datagen.flow(image,
                                            batch_size=batch_size,
                                            save_to_dir=image_folder_to_save,
                                            save_prefix=filename,
                                            save_format=save_format)

        # generate n samples
        count = 0
        for image in current_image_gen:  # accessing the object saves the image to disk
            count += 1
            if count == n:  # n images were generated
                break
        print('\tGenerate {} samples for file {}'.format(n, filename))
        i += 1

    print("\nTotal number images generated = {}".format(n*total))



for cla in class_names:
    image_folder_to_generate = os.path.join(image_folder, cla)
    image_folder_to_save = os.path.join(aug_image_folder, cla)
    origin_image_to_aug_image(image_folder_to_generate,image_folder_to_save, n)



