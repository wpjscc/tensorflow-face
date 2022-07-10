
## 第一步 分割图片

在datasets/face_dataset_train_images 下放一些自己的图片和非自己的图片

运行
```
python data_augmentation.py 
```

会在datasets/face_dataset_train_aug_images 文件夹下生成很多张图片

## 训练数据并保存模型数据

运行 
```
python train_and_save_data.py
```

会在 models 文件夹下生成 models/face_classifier.h5 文件

## 测试数据

在datasets/face_dataset_test_images 放置一些数据，用于测试

运行

```
python validation_model.py
```
会生成图片的百分比

## 使用摄像头验证

```
python video-validation_model.py
```