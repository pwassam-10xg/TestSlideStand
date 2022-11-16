from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread

import numpy as np
import logging
import sys
import pandas as pd

from TestSlideStand.ui_TestSlideStand import Ui_TestSlideStand

from TestSlideStand import TestSlideStand
from TestSlideStand.settings import settings
from pydispatch import dispatcher

class TestSlideWorkThread(QThread):
    def __init__(self, s: TestSlideStand):
        super().__init__()
        self.s = s

    def run(self):
        self.s.scan()

class TestSlideStandGUI(QMainWindow, Ui_TestSlideStand):
    frame = pyqtSignal(np.ndarray)
    exposure = pyqtSignal(float)
    angle = pyqtSignal(float)
    status = pyqtSignal(str)
    parallel = pyqtSignal(float)
    newdata = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.log = logging.getLogger(self.__class__.__name__)

        self.setupUi(self)
        self.s = TestSlideStand(settings=settings, ref='none')

        self.actionExit.triggered.connect(self.close)

        self.button_measure.clicked.connect(self.on_measure)
        dispatcher.connect(lambda x: self.frame.emit(x),    'FRAME',    weak=False)
        dispatcher.connect(lambda x: self.exposure.emit(x), 'EXPOSURE', weak=False)
        dispatcher.connect(lambda x: self.angle.emit(x),    'ANGLE',    weak=False)
        dispatcher.connect(lambda x: self.status.emit(x),   'STATUS',   weak=False)
        dispatcher.connect(lambda:   self.newdata.emit(),   'NEWDATA',  weak=False)
        dispatcher.connect(lambda x: self.parallel.emit(x), 'PARALLEL', weak=False)

        self.frame.connect(self.on_frame)
        self.exposure.connect(self.on_exposure)
        self.angle.connect(self.on_angle)
        self.status.connect(self.on_status)
        self.newdata.connect(self.on_newdata)
        self.parallel.connect(self.on_parallel)

        self.show()

    @pyqtSlot()
    def on_measure(self):
        print("Measure")
        self.thread = TestSlideWorkThread(self.s)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        self.log.info("Frame")
        self.iv.setImage(frame)
        self.log.info("Frame")

    @pyqtSlot(float)
    def on_exposure(self, exposure: float):
        self.label_exposure.setText(f'{exposure:.1f}')

    @pyqtSlot(float)
    def on_angle(self, angle: float):
        self.label_angle.setText(f'{angle:.1f}')

    @pyqtSlot(str)
    def on_status(self, status: str):
        self.statusbar.showMessage(status)

    @pyqtSlot()
    def on_newdata(self):
        data = self.s.analyzer._results
        df = pd.DataFrame(data.values(), index=data.keys())
        self.plots.plot_data(df)

    @pyqtSlot(float)
    def on_parallel(self, parallelism: float):
        self.label_parallelism.setText(f'{parallelism:.4f}')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(name)20s %(levelname)s %(message)s', level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    app = QApplication(sys.argv)
    win = TestSlideStandGUI()
    app.exec()
