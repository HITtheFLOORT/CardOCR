'''
使用训练好的模型进行预测
输入测试图片文件夹路径
获取最大可能标签
'''

from keras.models import load_model
import argparse
import pickle
import cv2
import os
# 设置输入参数

def predict_num(image_path,model,lb,path):

    size = 64,64
    flatten_val = -1


    # 加载测试数据并进行相同预处理操作
    image = cv2.imread(image_path)
    output = image.copy()
    image = cv2.resize(image, size)  # width=32,hight=32

    # 浮点
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

    if label[6] == path:
        return  1
    else:
        return  0
if __name__ == '__main__' :
    #测试图片路径和模型标签
    paths = "0","1","2","3","4","5","6","7","8","9"
    model_path = "../output_cnn/vggnet.model"
    label_path = "../output_cnn/vggnet_lb.pickle"
    # 读取模型和标签
    print("[INFO] loading network and label binarizer...")
    model = load_model(model_path)
    lb = pickle.loads(open(label_path, "rb").read())
    for path in paths:
        acc_percent = 0
        los_percent = 0
        pathDir = os.listdir(path)
        for allDir in pathDir:
            child = os.path.join(path, allDir)
            if os.path.isfile(child):
                if predict_num(child, model, lb, path) == 1:
                    acc_percent += 1
                else:
                    los_percent += 1
        #每次测试结果
        print("number" + path)
        print("{}: {:.2f}%".format((str)(len(pathDir))+"张测试图片准确率", acc_percent*100 / (acc_percent + los_percent)))
