# -*- coding: utf-8 -*-
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import sys
import os
import numpy as np
import utils
up_factor = 4

class Ui_MainWindow(PyQt5.QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.selection = 'Content-Loss'
        self.lowres_img = None
        self.lowres_input = None
        self.tmp = None
        self.filename = None
        self.prediction = None
        self.image = None
        self.model = None
        self.upfactor = 4

        

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1, 1)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.horizontalLayout_3.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.gridLayout_3.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusBar)

        self.combobox1 = QtWidgets.QComboBox(self.centralwidget)
        self.combobox1.setFixedSize(300,60)
        self.combobox1.move(30, -10)
        self.combobox1.addItem('RGB Content-Loss')
        self.combobox1.addItem('RGB MSE')
        self.combobox1.addItem('YCbCr Content-Loss')
        self.combobox1.addItem('YCbCr MSE')
        self.combobox1.addItem('RGB GAN Perceptual-Loss')
        self.combobox1.addItem('YCbCr GAN Perceptual-Loss')
        self.combobox1.currentTextChanged.connect(self.text_check)


        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton.clicked.connect(self.superres)
        self.pushButton_3.clicked.connect(self.bicubic)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.filename = None  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display

    def text_check(self, s):
        print("Text changed:", s)
        self.selection = s

    def loadImage(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = tf.keras.preprocessing.image.load_img(self.filename)
        self.image = tf.keras.preprocessing.image.img_to_array(self.image)
        img_array = self.image.astype("float32") / 255.0

        # Create a new figure with a default 111 subplot.
        fig, ax = plt.subplots()
        im = ax.imshow(img_array[::-1], origin="lower")

        plt.title('Ground Truth')
        plt.axis('off')
        # zoom-factor: 2.0, location: upper-left
        axins = zoomed_inset_axes(ax, 2, loc=2)
        axins.imshow(img_array[::-1], origin="lower")

        # Specify the limits.
        x1, x2, y1, y2 = 200, 300, 100, 200
        # Apply the x-limits.
        axins.set_xlim(x1, x2)
        # Apply the y-limits.
        axins.set_ylim(y1, y2)

        plt.yticks(visible=False)
        plt.xticks(visible=False)

        # Make the line.
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
        plt.show()


    def superres(self):

        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = tf.keras.preprocessing.image.load_img(self.filename)
        self.lowres_input = utils.get_lowres_image(self.image, self.upfactor)

        if self.selection == 'RGB Content-Loss':
            self.model = keras.models.load_model('models/img_models/rgb_crop_CL_50_x4.h5', compile=False)
            channel = 'rgb'

        if self.selection == 'YCbCr Content-Loss':
            self.model = keras.models.load_model('models/img_models/ycbcr_crop_CL_50_x4.h5', compile=False)
            channel = "ycbcr"

        if self.selection == 'YCbCr MSE':
            self.model = keras.models.load_model('models/img_models/ycbcr_crop_MSE_50_x4.h5', compile=False)
            channel = "ycbcr"

        if self.selection == 'RGB MSE':
            self.model = keras.models.load_model('models/img_models/rgb_crop_MSE_50_x4.h5', compile=False)
            channel = "rgb"

        if self.selection == 'RGB GAN Perceptual-Loss':
            self.model = keras.models.load_model('models/img_models/generator_x4.h5', compile=False)
            channel = "rgb"

        if self.selection == 'YCbCr GAN Perceptual-Loss':
            channel = 'ycbcr'
            pass

        self.prediction = utils.upscale_image(self.model, self.lowres_input, channels=channel)
        print(self.prediction)
        img_array = tf.keras.preprocessing.image.img_to_array(self.prediction)
        img_array = img_array.astype("float32") / 255.0

        # Create a new figure with a default 111 subplot.
        fig, ax = plt.subplots()
        im = ax.imshow(img_array[::-1], origin="lower")

        plt.title('SuperRezzed with '+self.selection)
        plt.axis('off')
        # zoom-factor: 2.0, location: upper-left
        axins = zoomed_inset_axes(ax, 2, loc=2)
        axins.imshow(img_array[::-1], origin="lower")

        # Specify the limits.
        x1, x2, y1, y2 = 200, 300, 100, 200
        # Apply the x-limits.
        axins.set_xlim(x1, x2)
        # Apply the y-limits.
        axins.set_ylim(y1, y2)

        plt.yticks(visible=False)
        plt.xticks(visible=False)

        # Make the line.
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")

        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        plt.savefig(filename+'superrezzed'+'.png')
        plt.show()


    def bicubic(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = tf.keras.preprocessing.image.load_img(self.filename)
        self.lowres_input = utils.get_lowres_image(self.image, self.upfactor)
        w = self.lowres_input.size[0] * up_factor
        h = self.lowres_input.size[1] * up_factor
        self.lowres_img = self.lowres_input.resize((w, h))
        img_array = tf.keras.preprocessing.image.img_to_array(self.lowres_img)
        img_array = img_array.astype("float32") / 255.0
        # Create a new figure with a default 111 subplot.
        fig, ax = plt.subplots()
        im = ax.imshow(img_array[::-1], origin="lower")

        plt.title('Bicubic')
        plt.axis('off')
        # zoom-factor: 2.0, location: upper-left
        axins = zoomed_inset_axes(ax, 2, loc=2)
        axins.imshow(img_array[::-1], origin="lower")

        # Specify the limits.
        x1, x2, y1, y2 = 200, 300, 100, 200
        # Apply the x-limits.
        axins.set_xlim(x1, x2)
        # Apply the y-limits.
        axins.set_ylim(y1, y2)

        plt.yticks(visible=False)
        plt.xticks(visible=False)

        # Make the line.
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")

        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        plt.savefig(filename + 'Bicubic' + '.png')
        plt.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image SuperRezzer"))
        self.pushButton_2.setText(_translate("MainWindow", "Open"))
        self.pushButton.setText(_translate("MainWindow", "SuperRes Image"))
        self.pushButton_3.setText(_translate("MainWindow", "Bicubic Image"))






if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())




