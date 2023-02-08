import re
from pathlib import PurePath

from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5
from PyQt5.QtCore import QUrl
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

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        #self.gridLayout_3.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        # spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusBar)


        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(self.audio_multi)
        self.pushButton.clicked.connect(self.audio_single)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Added code here
        self.filename = None  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display

    def audio_single(self):
        self.filename = QFileDialog.getOpenFileName()[0]
        utils.audio(self.filename, type="single")

    def audio_multi(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        utils.audio(self.filename, type="multi")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Audio SuperRezzer"))
        self.pushButton_2.setText(_translate("MainWindow", "Multi Speaker"))
        self.pushButton.setText(_translate("MainWindow", "Single Speaker"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())