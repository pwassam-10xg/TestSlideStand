from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread

import numpy as np
import logging
import pandas as pd

from TestSlideStand.ui_TestSlideStand import Ui_TestSlideStand

from TestSlideStand import TestSlideStand
from TestSlideStand import __version__ as TestSlideStandversion
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
    final_angular = pyqtSignal(float)
    final_spatial = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread: Optional[TestSlideWorkThread] = None
        self.log = logging.getLogger(self.__class__.__name__)

        self.setupUi(self)
        self.setWindowTitle(f'TestSlideStand {TestSlideStandversion}')
        self.s = TestSlideStand(settings=settings, ref='none')
        self.ref: Optional[str] = None

        self.actionExit.triggered.connect(self.close)
        self.actionSave.triggered.connect(self.on_save)
        self.actionLoad.triggered.connect(self.on_load)

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
        self.final_angular.connect(self.on_final_angular)
        self.final_spatial.connect(self.on_final_spatial)

        self.button_measure.setDisabled(True)
        self.lineEdit_ref.textChanged.connect(self.on_ref)
        self.show()

    @pyqtSlot()
    def on_save(self):
        if not self.ref:
            self.statusbar.showMessage("No reference")
            return

        if self.thread and self.thread.isRunning():
            self.statusbar.showMessage('Cant save while measuring')
            return

        dialog = SaveDialog()
        dirname = dialog.getSaveFileName(directory=str(settings.datadir))
        if dirname:
            dirname = Path(dirname[0])
            dirname.mkdir(parents=True, exist_ok=True)
            self.s.analyzer.save(dirname, ref=self.ref)

    @pyqtSlot()
    def on_load(self):
        pass

    @pyqtSlot()
    def on_ref(self):
        ret = self.lineEdit_ref.text().strip()
        if ret == '':
            self.ret = None
            self.button_measure.setDisabled(True)
        else:
            self.ret = ret
            self.button_measure.setEnabled(True)

    @pyqtSlot()
    def on_measure(self):
        if self.thread and self.thread.isRunning():
            self.statusbar.showMessage('Already measuring')
        else:
            self.lineEdit_ref.setDisabled(True)
            self.ref = self.lineEdit_ref.text().strip()
            self.button_measure.setDisabled(True)
            self.s.analyzer.reset()
            self.plots.reset()
            self.thread = TestSlideWorkThread(self.s)
            self.thread.start()
            self.thread.finished.connect(self.on_measure_finished)

    @pyqtSlot()
    def on_measure_finished(self):
        self.button_measure.setDisabled(True)
        self.lineEdit_ref.clear()
        self.lineEdit_ref.setEnabled(True)

    @pyqtSlot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        self.log.info("Frame")
        self.iv.setImage(np.flipud(frame))
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

    @pyqtSlot(float)
    def on_final_angular(self, v: float):
        self.label_angle_dep.setText(f'{v:.3f} %')

    @pyqtSlot(float)
    def on_final_spatial(self, v: float):
        self.label_spatial_dep.setText(f'{v:.3f} %')

class SaveDialog(QFileDialog):
    def __init__(self):
        super().__init__()

        self.FileMode = QFileDialog.FileMode.Directory
        self.Options = QFileDialog.Option.ShowDirsOnly
        self.ViewMode = QFileDialog.ViewMode.Detail
