import matplotlib
from model_name.simple_vggnet import SimpleVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from my_utils import utils_paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# 设置参数

dataset="../dataset"
model_path="../output_cnn/vggnet.model"
lb_path="../output_cnn/vggnet_lb.pickle label"
image_output="../output_cnn/vggnet_plot.png"
# 读取数据和标签
print("[INFO] 读取图片...")
data = []
labels = []

# 拿到路径
imagePaths = sorted(list(utils_paths.list_images(dataset)))
random.seed(66)
random.shuffle(imagePaths)

# 读取数据 size 64x64
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# 预处理
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 数据集切分
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# 标签转换
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 数据增强

aug = ImageDataGenerator(rotation_range=10,
						 width_shift_range=0.1,
						 height_shift_range=0.1,
						 shear_range=0.2,
						 zoom_range=0.2,
						 horizontal_flip=False,
						 fill_mode="nearest")



# 建立卷积神经网络
model = SimpleVGGNet.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

# 初始化超参数
INIT_LR = 0.01
EPOCHS = 500
BS = 32

# 损失函数
print("[INFO] 训练网络...")
opt = SGD(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 训练网络

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)


'''
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)
'''




# 测试
print("[INFO] 测试网络...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# 展示结果
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N[:50], H.history["loss"], label="train_loss")
plt.plot(N[:50], H.history["val_loss"], label="val_loss")
plt.plot(N[:50], H.history["acc"], label="train_acc")
plt.plot(N[:50], H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy ")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(image_output)

# 保存模型
print("[INFO] 保存模型...")
model.save(model_path)
f = open(lb_path, "wb")
f.write(pickle.dumps(lb))
f.close()