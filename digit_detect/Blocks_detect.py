
import pickle

import cv2
from keras.models import load_model

from digit_detect.Image_recognize import ImageRecongnize
'''
纵轴滑动窗口获取图相框并识别
要求card尽量占满屏幕上下沿
'''

class BlockDetection:
    def blockdetect(self,image):
        model = load_model("E:/workspacecourse/Keras_learn/output_cnn/vggnet.model")
        lb = pickle.loads(open("E:/workspacecourse/Keras_learn/output_cnn/vggnet_lb.pickle", "rb").read())
        orig = image.copy()
        (H, W) = image.shape[:2]
        IR = ImageRecongnize(model, lb)
        a = (int)(0.105 * H)#0.10
        i = (int)(0.4 * H)
        s = ""
        roi_begin = 0
        roi_end = 0
        num=0
        while i >= (int)(0.4 * H) and i < (int)(0.7 * H):
            block = image[i:i + a, 0:W]
            st, begin, end, snum = IR.recognize(block)
            if len(st) > len(s):
                #cv2.putText(orig, "result:" + st, (begin, i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(orig, (begin, i), (end + (int)(0.5 * a), i + a), (0, 0, 255), 2)
                s = st
                num = snum
                roi_begin = begin
                roi_end = end
            if len(st) > 0 and len(st) <= len(s):
                break
            i += (int)(0.25 * a)
        return s,orig#位置图片框选
    def blockdetect_pos(self,image):
        model = load_model("E:/workspacecourse/Keras_learn/output_cnn/vggnet.model")
        lb = pickle.loads(open("E:/workspacecourse/Keras_learn/output_cnn/vggnet_lb.pickle", "rb").read())
        orig = image.copy()
        (H, W) = image.shape[:2]
        IR = ImageRecongnize(model, lb)
        a = (int)(0.125 * H)#0.10
        i = (int)(0.4 * H)
        s = ""

        while i >= (int)(0.4 * H) and i < (int)(0.7 * H):
            block = image[i:i + a, 0:W]
            st, begin, end, snum = IR.recognize(block)
            if len(st) > len(s):
                #cv2.putText(orig, "result:" + st, (begin, i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(orig, (begin, i), (end + (int)(0.5 * a), i + a), (0, 0, 255), 2)
                s = st

            if len(st) > 0 and len(st) <= len(s):
                break
            i += (int)(0.25 * a)
        return s,(begin,end+(int)(0.5 * a),i,i+a) #位置信息

if __name__ == '__main__':
    file_path = "../test_images/image(48).jpg"
    image=cv2.imread(file_path)
    br=BlockDetection()
    s,result=br.blockdetect(image)
    print(s)
    cv2.imshow("handhold image",result)
    cv2.waitKey(0)