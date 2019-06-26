
import numpy as np
import cv2
from keras.models import load_model
import pickle
'''
横轴滑动窗口，已做部分截断
纵横比代表单字符宽度所占长的比利
可选择限制文本一识别不同类型的文本条目
优化数字1处理效果与0，6相矛盾
'''
class ImageRecongnize:
    def __init__(self,model,lb):
        self.model = model#模型
        self.lb = lb#标签
        self.size = 64, 64#model的size
        self.specific_value = 0.50#水平滑块纵横比
        self.judge_value = 90#判别阈值
        self.digit_restriction = True
    def predeal_predict(self,im):
        im = cv2.resize(im, self.size)
        # 图形浮点数，通道处理
        im = im.astype("float") / 255.0
        im = im.reshape((1, im.shape[0], im.shape[1],
                         im.shape[2]))
        # 预测
        preds = self.model.predict(im)
        # 得到预测结果以及其对应的标签
        num = preds.argmax(axis=1)[0]
        label = self.lb.classes_[num]
        return preds,num,label
    def recognize(self,image):
        number = ""
        (H, W) = image.shape[:2]
        a = (int)(H * self.specific_value)
        i = 0
        begin = 0
        end = W
        snum=0
        while i + a < W:
            #滑动窗口区域，少于一定字符则不是银行卡号
            if i > 0.3*W and snum < 2:
                break
            if i > 0.5*W and snum < 5:
                break
            if i > 0.7*W and snum < 10:
                break
            im = image[0:H, i:i + a]
            preds,num,label=self.predeal_predict(im)
            if preds[0][num] * 100 > self.judge_value:
                # cv2.imshow("re", im2)
                # cv2.waitKey(0)
                if i + a < W:
                    # print(label, preds[0][num], i)
                    if not label[6:] == "10":
                        if label[6:] == "1":
                            f=(int)(0.25 * a)
                            im = image[0:H, i+f:i + a + f]
                            preds, num, label = self.predeal_predict(im)
                            i += f
                        number += label[6:]
                        snum += 1
                        if begin == 0:
                            begin = i
                        end = i + a
                        i += a
                    else:
                        if not len(number)==0 and not number[-1] == " ":
                            number += " "
                        i += (int)(a * 0.15)
            else:
                i += (int)(a * 0.15)
        if snum < 14 and self.digit_restriction == True:
            number=""
        #识别结果，开始位置，结束位置，数字数目
        return number,begin,end,snum

if __name__=='__main__':
    image = cv2.imread("../dataset2/number/long8.png")
    model = load_model("../output_cnn/vggnet.model")
    lb = pickle.loads(open("../output_cnn/vggnet_lb.pickle", "rb").read())
    IR=ImageRecongnize(model,lb)
    s, begin, end, snum = IR.recognize(image)
    print(s)
    cv2.imshow(s,image)
    cv2.waitKey(0)


