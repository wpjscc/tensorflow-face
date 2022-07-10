# %% [markdown]
# # Import Libraries

# %%
# Common imports
import os
import numpy as np

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# TensorFlow imports
# may differs from version to versions

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing import image

# 训练文件夹
# me/not_me
# 训练的模型会根据子文件夹名称作为数一类据的key
train_image_folder = os.path.join('datasets', 'face_dataset_train_aug_images')
img_height, img_width = 250, 250  # size of images
num_classes = 2  # me - not_me

# Training settings
validation_ratio = 0.15  # 15% for the validation
batch_size = 16

AUTOTUNE = tf.data.AUTOTUNE

model_lujiing =  'models/face_classifier.h5'


# 训练 subset="training"
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_image_folder,
    validation_split=validation_ratio,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    label_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

class_names = train_ds.class_names
print(class_names)
#['me', 'not_me']


## 验证 subset="validation"--可以理解为，每次训练后用源数据去验证，是否正确
val_ds = keras.preprocessing.image_dataset_from_directory(
    train_image_folder,
    validation_split=validation_ratio,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True)


##  使用ResNet50模型
base_model = keras.applications.ResNet50(weights='imagenet',
                                         include_top=False,  # without dense part of the network
                                         input_shape=(img_height, img_width, 3))
# 不带顶部的模型
#base_model.summary()

# Set layers to non-trainable
for layer in base_model.layers:
    layer.trainable = False



# 添加自定义层--全局平均池化层和密集层
global_avg_pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(num_classes, activation='sigmoid')(global_avg_pooling)

face_classifier = keras.models.Model(inputs=base_model.input,
                                     outputs=output,
                                     name='ResNet50')
face_classifier.summary()

# 训练的时候通过的两个回调函数-大概意思是将每一次训练后的数据放在 model_lujiing（还没落到磁盘上）
checkpoint = ModelCheckpoint(model_lujiing,
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

#EarlyStopping作为一种正则化技术（不太明白）---大概意思是3次后不会提升训练度将停止训练
earlystop = EarlyStopping(monitor='val_loss',
                          restore_best_weights=True,
                          patience=3,  # number of epochs with no improvement after which training will be stopped
                          verbose=1)

# 将两个放在一起---判断训练度有没有提升earlystop，提升的话继续训练checkpoint
callbacks = [earlystop, checkpoint]

# 使用具有标准学习率值的-Adam
face_classifier.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(learning_rate=0.01),
                        metrics=['accuracy'])

# 开始训练完成后将数据保存在models/face_classifier.h5
# 训练50次，直到训练度没有提升
epochs = 50
# %%
history = face_classifier.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds)

# 保存模型数据
face_classifier.save(model_lujiing)




