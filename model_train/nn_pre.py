# 导入所需工具包
from keras.models import load_model
import argparse
import pickle
import cv2
import os
# 设置输入参数

def predict_num(image_path,model,lb,path):
    size = 32, 32
    flatten_val = 1
    # 加载测试数据并进行相同预处理操作
    image = cv2.imread(image_path)
    output = image.copy()
    image = cv2.resize(image, size)  # width=32,hight=32

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

    # 预测
    preds = model.predict(image)

    # 得到预测结果以及其对应的标签
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    # 在图像中把结果画出来
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    # cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    #	(0, 0, 255), 2)
    #print(text)

    if label[6] == path:
        return  1
    else:
        return  0
if __name__ == '__main__' :
    paths = "0","1","2","3","4","5","6","7","8","9"#single_digit_test路径下单字符识别率

    model_path = "../output/simple_nn.model"
    label_path = "../output/simple_nn_lb.pickle"
    # 读取模型和标签
    print("[INFO] 读取模型...")
    model = load_model(model_path)
    lb = pickle.loads(open(label_path, "rb").read())
    for path in paths:
        acc_percent = 0
        los_percent = 0
        pathDir = os.listdir(path)
        for allDir in pathDir:
            child = os.path.join(path, allDir)
            if os.path.isfile(child):
                if predict_num(child,model,lb,path) == 1:
                    acc_percent+= 1
                else:
                    los_percent+= 1

        print("number"+path)
        print(acc_percent/(acc_percent+los_percent))