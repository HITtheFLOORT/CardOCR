import os
import cv2
'''
数据集生成，对原始四位数字图例提取
'''

# 遍历指定目录，显示目录下的所有文件名
def CropImage4File(filepath,charar):
    destpath = 'image'
    pathDir = os.listdir(filepath)  # 列出文件路径中的所有路径或文件
    index=0
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        if os.path.isfile(child):
            image = cv2.imread(child)

        ch = charar
        destpath0=destpath+'_'+ch
        if child[7]==ch:
            cropImg = image[0:46, 0:30]  # 裁剪图像
            imagename = child[7] + '_' + str(index) + ".png"
            print(imagename)
            cv2.imwrite(destpath0 + '/' + imagename, cropImg)  # 写入图像路径
            index += 1;
        if child[8]==ch:
            cropImg2 = image[0:46, 30:60]  # 裁剪图像
            imagename = child[8] + '_' + str(index) + ".png"
            print(imagename)
            cv2.imwrite(destpath0 + '/' + imagename, cropImg2)  # 写入图像路径
            index += 1;
        if child[9] == ch:
            cropImg3 = image[0:46, 60:90]  # 裁剪图像
            imagename = child[9] + '_' + str(index) + ".png"
            print(imagename)
            cv2.imwrite(destpath0 + '/' + imagename, cropImg3)  # 写入图像路径
            index += 1;
        if child[10] == ch:
            cropImg4 = image[0:46, 90:120]  # 裁剪图像
            imagename = child[10] + '_' + str(index) + ".png"
            print(imagename)
            cv2.imwrite(destpath0 + '/' + imagename, cropImg4)  # 写入图像路径
            index += 1;
        if index > 1000:
            break

if __name__ == '__main__':
    filepath = 'images'  # 源图像
    CropImage4File(filepath,"9")#截取图像
