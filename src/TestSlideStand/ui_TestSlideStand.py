# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_TestSlideStand.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TestSlideStand(object):
    def setupUi(self, TestSlideStand):
        TestSlideStand.setObjectName("TestSlideStand")
        TestSlideStand.resize(1160, 770)
        self.centralwidget = QtWidgets.QWidget(TestSlideStand)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_parallelism = QtWidgets.QLabel(self.centralwidget)
        self.label_parallelism.setMinimumSize(QtCore.QSize(100, 0))
        self.label_parallelism.setText("")
        self.label_parallelism.setObjectName("label_parallelism")
        self.gridLayout.addWidget(self.label_parallelism, 1, 9, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 1, 1, 1)
        self.button_measure = QtWidgets.QPushButton(self.centralwidget)
        self.button_measure.setObjectName("button_measure")
        self.gridLayout.addWidget(self.button_measure, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(674, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 10, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 8, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 3, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.iv = ImageView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.iv.sizePolicy().hasHeightForWidth())
        self.iv.setSizePolicy(sizePolicy)
        self.iv.setObjectName("iv")
        self.gridLayout_2.addWidget(self.iv, 0, 0, 1, 1)
        self.plots = MplWidget(self.centralwidget)
        self.plots.setObjectName("plots")
        self.gridLayout_2.addWidget(self.plots, 0, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 3)
        self.gridLayout_2.setColumnStretch(1, 2)
        self.gridLayout.addLayout(self.gridLayout_2, 2, 0, 1, 12)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 6, 1, 1)
        self.label_angle = QtWidgets.QLabel(self.centralwidget)
        self.label_angle.setMinimumSize(QtCore.QSize(100, 0))
        self.label_angle.setText("")
        self.label_angle.setObjectName("label_angle")
        self.gridLayout.addWidget(self.label_angle, 1, 7, 1, 1)
        self.label_exposure = QtWidgets.QLabel(self.centralwidget)
        self.label_exposure.setMinimumSize(QtCore.QSize(100, 0))
        self.label_exposure.setText("")
        self.label_exposure.setObjectName("label_exposure")
        self.gridLayout.addWidget(self.label_exposure, 1, 5, 1, 1)
        self.lineEdit_ref = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_ref.setMinimumSize(QtCore.QSize(100, 0))
        self.lineEdit_ref.setObjectName("lineEdit_ref")
        self.gridLayout.addWidget(self.lineEdit_ref, 1, 2, 1, 1)
        TestSlideStand.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TestSlideStand)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1160, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        TestSlideStand.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(TestSlideStand)
        self.statusbar.setObjectName("statusbar")
        TestSlideStand.setStatusBar(self.statusbar)
        self.actionSave = QtWidgets.QAction(TestSlideStand)
        self.actionSave.setObjectName("actionSave")
        self.actionExit = QtWidgets.QAction(TestSlideStand)
        self.actionExit.setObjectName("actionExit")
        self.actionAbout = QtWidgets.QAction(TestSlideStand)
        self.actionAbout.setObjectName("actionAbout")
        self.actionLoad = QtWidgets.QAction(TestSlideStand)
        self.actionLoad.setObjectName("actionLoad")
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExit)
        self.menuFile.addSeparator()
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(TestSlideStand)
        QtCore.QMetaObject.connectSlotsByName(TestSlideStand)

    def retranslateUi(self, TestSlideStand):
        _translate = QtCore.QCoreApplication.translate
        TestSlideStand.setWindowTitle(_translate("TestSlideStand", "TestSlideStand"))
        self.label_4.setText(_translate("TestSlideStand", "Slide ID:"))
        self.button_measure.setText(_translate("TestSlideStand", "Measure"))
        self.label_3.setText(_translate("TestSlideStand", "Parallelism: "))
        self.label.setText(_translate("TestSlideStand", "Exposure (uS): "))
        self.label_2.setText(_translate("TestSlideStand", "Angle (deg): "))
        self.menuFile.setTitle(_translate("TestSlideStand", "File"))
        self.menuHelp.setTitle(_translate("TestSlideStand", "Help"))
        self.actionSave.setText(_translate("TestSlideStand", "Save"))
        self.actionExit.setText(_translate("TestSlideStand", "Exit"))
        self.actionAbout.setText(_translate("TestSlideStand", "About"))
        self.actionLoad.setText(_translate("TestSlideStand", "Load"))
from mplwidget import MplWidget
from pyqtgraph import ImageView
