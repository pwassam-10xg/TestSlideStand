import pydispatch.dispatcher
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QStyle, QComboBox, QShortcut, QLineEdit, \
    QPushButton, QFileDialog, QLabel
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QSettings, QPointF, QPoint, QObject

import numpy as np
import logging
import matplotlib
import sys
matplotlib.use('Qt5Agg')

from TestSlideStand.ui_TestSlideStand import Ui_TestSlideStand

from TestSlideStand import TestSlideStand
from TestSlideStand.settings import settings
from pydispatch import dispatcher

class TestSlideWorkThread(QObject):
    def __init__(self, s: TestSlideStand):
        super().__init__()
        self.s = s

    @pyqtSlot()
    def measure(self):
        self.s.scan()

class TestSlideStandGUI(QMainWindow, Ui_TestSlideStand):
    frame = pyqtSignal(np.ndarray)
    exposure = pyqtSignal(float)
    angle = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.log = logging.getLogger(self.__class__.__name__)

        self.setupUi(self)

        self.worker = TestSlideWorkThread(TestSlideStand(settings=settings, ref='none'))

        self.actionExit.triggered.connect(self.close)
        self.button_acquire.connect(self.worker.measure)

        dispatcher.connect('FRAME', lambda x: self.frame.emit(x))
        dispatcher.connect('EXPOSURE', lambda x: self.exposure.emit(x))
        dispatcher.connect('ANGLE', lambda x: self.angle.emit(x))

        self.show()
        print("Inited")

    @pyqtSlot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        pass

    @pyqtSlot(float)
    def on_exposure(self, exposure: float):
        pass

    @pyqtSlot(float)
    def on_angle(self, angle: float):
        pass


if __name__ == '__main__':
    print("Start")
    app = QApplication(sys.argv)
    win = TestSlideStandGUI()
    app.exec()
    print("End")