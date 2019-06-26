import os
import cv2
from digit_detect.Blocks_detect import BlockDetection

filepath="test_image"#测试文件夹
destpath="test_result"#结果文件夹
pathDir = os.listdir(filepath)
index=0
br = BlockDetection()

i=40

re=""

for allDir in pathDir:
    child = os.path.join(filepath, allDir)
    if os.path.isfile(child):
        image = cv2.imread(child)
        s, result = br.blockdetect(image)
        cv2.imwrite(destpath + "/" +(str)(i)+".png",result)
        print((str)(i)+" image complete")
        re+="card_"+(str)(i)+" "+s+"\n"
        i+=1
    if i>50:
        break
fw = open(destpath + "/" + "result.txt", 'a')
fw.writelines(re)
fw.close()