import pickle
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from keras.models import load_model
from digit_detect.Image_recognize import ImageRecongnize
'''
east文本定位，没有卡片训练集
使用原始模型，效果对于部分卡片较差
'''

class eastRoi:
    def __init__(self,modelpath,imagepath):
        self.width = 736
        self.height = 384
        self.min_confidence = 0.5
        self.modelpath = modelpath
        self.imagepath = imagepath
    def build(self):
        # load the input image and grab the image dimensions
        image = cv2.imread(self.imagepath)
        orig = image.copy()
        (H, W) = image.shape[:2]

        # set the new self.width and height and then determine the ratio in change
        # for both the self.width and height
        (newW, newH) = (self.width,self.height)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        orig = image.copy()
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(self.modelpath)

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)

        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # loading model
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < self.min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the self.width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                if startY < (int)(0.5 * self.height) or startY > (int)(0.65 * self.height):
                    continue
                #if endX - startX < 100:
                    #continue
                if endY - startY < 30:
                    continue
                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # loop over the bounding boxes
        size = 64, 64

        for i in range(len(boxes) - 1):
            for j in range(i + 1, len(boxes)):
                (startX, startY, endX, endY) = boxes[i]
                (startx, starty, endx, endy) = boxes[j]
                if startX > startx:
                    (x, y, z, c) = boxes[j]
                    boxes[j] = boxes[i]
                    boxes[i] = (x, y, z, c)
        '''
                if len(boxes) < 4:
            minn = self.width
            for i in range(len(boxes) - 1):
                (startX, startY, endX, endY) = boxes[i]
                (startx, starty, endx, endy) = boxes[i+1]
                if startx-startX <minn:
                    minn= startx-startX
                    print(minn)
            for i in range(len(boxes) - 1):
                (startX, startY, endX, endY) = boxes[i]
                (startx, starty, endx, endy) = boxes[i+1]
                if startx-startX >1.5*minn:
                    boxes.insert((startX+minn, startY, endX+minn, endY),i)
        '''

        for i in range(len(boxes)):
            (startX, startY, endX, endY) = boxes[i]
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
            #cv2.putText(orig, (str)(i), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return boxes,image,orig
    def boxesDetect(self):
        boxes,image,orig=self.build()
        model = load_model("E:/workspacecourse/Keras_learn/output_cnn/vggnet.model")
        lb = pickle.loads(open("E:/workspacecourse/Keras_learn/output_cnn/vggnet_lb.pickle", "rb").read())
        IR=ImageRecongnize(model,lb)
        IR.specific_value=0.60
        s=""
        for i in range(len(boxes)):
            (startX, startY, endX, endY) = boxes[i]
            number, begin, end, snum=IR.recognize(image[startY:endY,startX:endX])
            #cv2.imshow(number,image[startY:endY,startX:endX])
            #cv2.waitKey(0)
            s+=number
        #print(s)
        return s

if __name__=='__main__':
    eR=eastRoi("frozen_east_text_detection.pb","../test_image/test3.png")
    boxes, image, orig=eR.build()
    cv2.imshow("re",orig)
    cv2.waitKey(0)


