#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: feifan time:2019/5/26
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog,QLabel,QWidget
from PyQt5.QtGui import *
from mainWindows import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from QCandyUi import CandyWindow
import cv2
from digit_detect.Blocks_detect import BlockDetection


class DetailUI(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(DetailUI, self).__init__()
        self.setupUi(self)

        self.setWindowOpacity(0.95)   #设置窗口透明度
        self.setAttribute(Qt.WA_SetStyle)#设置窗口透明
        QMainWindow.setStyleSheet(self,"background-color:blue;")   # 设置背景颜色
        QMainWindow.setStyleSheet(self,"border-image: url(:/image/background.png);") # 设置背景图片
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_8.setSpacing(0)
        self.lineEdit_url.setStyleSheet('''
        QLineEdit{
            border:1px solid gray;
            width:300px;
            border-radius:10px;
            padding:2px 4px;
            }''')
        self.lineEdit_bugNumber.setStyleSheet('''
                QLineEdit{
                    border:1px solid gray;
                    width:300px;
                    border-radius:10px;
                    padding:2px 4px;
                    }''')

    def openImage(self):
        self.openfile = QFileDialog.getOpenFileName(self.centralwidget, 'open file', './test_image', "Image files (*.jpg *.gif *.png)")[0]
        self.lineEdit_url.setText(self.openfile)
        self.Im = cv2.imread(self.openfile)
        pix = QPixmap(self.openfile).scaled(self.label_init.width(), self.label_init.height() * 8 / 11)
        self.label_init.setPixmap(pix)
        self.label_bugImage.setPixmap(pix)

    def runModel(self):
        try:
            br = BlockDetection()
            s, self.Re = br.blockdetect(self.Im)
            image_height, image_width, image_depth = self.Re.shape  # 获取图像的高，宽以及深度。
            QIm = cv2.cvtColor(self.Re, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
            QIm = QImage(QIm.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                         image_width * image_depth,
                         QImage.Format_RGB888)
            pix = QIm.scaled(self.label_init.width(), self.label_init.height() * 8 / 11)
            self.label_init.setPixmap(QPixmap.fromImage(pix))
            self.resultShow(s)
        except:
            pass
    def resultShow(self,s):
        bg=cv2.imread("image/cardbg.png")
        cv2.putText(bg, "bank card", (20, 35), cv2.FONT_HERSHEY_COMPLEX , 1, (255, 255, 255) , 2)
        cv2.putText(bg, "Card Number:", (88, 200), cv2.FONT_HERSHEY_COMPLEX , 1, (255, 255, 255) , 2)
        cv2.putText(bg, s, (88, 260), cv2.FONT_HERSHEY_COMPLEX , 1, (255, 255, 255) , 2)
        image_height, image_width, image_depth = bg.shape  # 获取图像的高，宽以及深度。
        QIm = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
        QIm = QImage(QIm.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                     image_width * image_depth,
                     QImage.Format_RGB888)
        pix = QIm.scaled(self.label_out.width(), self.label_out.height() * 8 / 11)
        self.label_out.setPixmap(QPixmap.fromImage(pix))
        self.textEdit_advice.setText(self.openfile+'\n'+'card_number:'+s)
    def paraImage1(self):
        pix = QPixmap("model_name/vggnet_plot_nodecay_001.png")#.scaled(self.label_init.width(), self.label_init.height() * 8 / 11)
        msg = QPixmap(
            "model_name/nodecay.png")
        self.label_paraImage.setPixmap(pix)
        self.label_paraImage.setScaledContents(True)
        self.label_para.setPixmap(msg)
        self.label_para.setScaledContents(True)
    def paraImage2(self):
        pix = QPixmap(
            "model_name/vggnet_plot_decay.png")  # .scaled(self.label_init.width(), self.label_init.height() * 8 / 11)
        msg=QPixmap(
            "model_name/decay.png")
        self.label_paraImage.setPixmap(pix)
        self.label_paraImage.setScaledContents(True)
        self.label_para.setPixmap(msg)
        self.label_para.setScaledContents(True)
    def paraImage3(self):
        pix = QPixmap(
            "model_name/nn_plot.png")  # .scaled(self.label_init.width(), self.label_init.height() * 8 / 11)
        msg = QPixmap(
            "model_name/nn.png")
        self.label_paraImage.setPixmap(pix)
        self.label_paraImage.setScaledContents(True)
        self.label_para.setPixmap(msg)
        self.label_para.setScaledContents(True)
    def paraImage4(self):
        pix = QPixmap(
            "output_cnn/vggnet_plot_decay.png")  # .scaled(self.label_init.width(), self.label_init.height() * 8 / 11)
        msg = QPixmap(
            "model_name/decay.png")
        self.label_paraImage.setPixmap(pix)
        self.label_paraImage.setScaledContents(True)
        self.label_para.setPixmap(msg)
        self.label_para.setScaledContents(True)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DetailUI()
    ex = CandyWindow.createWindow(ex, 'blueGreen')
    QMainWindow.setWindowTitle(ex, "银行卡识别系统")
    ex.show()
    sys.exit(app.exec_())