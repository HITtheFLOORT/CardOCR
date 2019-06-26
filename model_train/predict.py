# 导入所需工具包
from keras.models import load_model
import argparse
import pickle
import cv2

# 设置输入参数

image_path = "../dataset2/number/3.png"
size = 64,64
flatten_val = -1
model_path = "../output_cnn/vggnet.model"
label_path = "../output_cnn/vggnet_lb.pickle"


# 加载测试数据并进行相同预处理操作
image = cv2.imread(image_path)
output = image.copy()
image = cv2.resize(image, size)#width=32,hight=32

# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0
flatten = flatten_val
# 是否要对图像就行拉平操作
if flatten > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))

# CNN的时候需要原始图像
else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

# 读取模型和标签
print("[INFO] 读取模型...")
model = load_model(model_path)
lb = pickle.loads(open(label_path, "rb").read())
# 预测
preds = model.predict(image)

# 得到预测结果以及其对应的标签
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)

print(text)




