__version__ = '0.0.1'

import logging
import numpy as np
import png

from typing import Dict
from dataclasses import dataclass
from pathlib import Path
from pydispatch import dispatcher

from TestSlideStand.camera import Cam
from TestSlideStand.stage import Stage
from TestSlideStand.analysis import ImageAnalyzer

__all__ = ['TestSlideStand']


class TestSlideStand:
    @dataclass
    class Settings:
        zaber_sn: str                                    # FTDI serial number of zaber
        positions: np.ndarray = np.arange(-30, 30, 2)    # Positions to scan
        auto_exp_pos: float = -30                        # Auto exposure position in degrees
        datadir: Path = Path(r'c:\TestSlideStandData')

    def __init__(self, settings: Settings, ref: str):
        self.log = logging.getLogger(self.__class__.__name__)
        self.settings = settings
        self.log.info("Init")

        self.zaber = Stage(settings.zaber_sn)
        self.cam = Cam()
        self.data: Dict[float, np.ndarray] = {}
        self.analyzer = ImageAnalyzer()

    def shutdown(self):
        self.zaber.shutdown()
        self.cam.shutdown()
        return

    def status(self, msg):
        self.log.info(msg)
        dispatcher.send('STATUS', dispatcher.Any, msg)

    def scan(self):
        self.status('Auto exposure')

        self.zaber.abs(self.settings.auto_exp_pos)
        exp = self.cam.auto_exposure()
        self.log.info("Auto exposure at %.1f uS", exp)

        for p in self.settings.positions:
            self.status(f"Scanning position {p:.1f}")
            self.zaber.abs(p)
            img = self.cam.snap()
            self.analyzer.add_image(p, img)

        self.zaber.abs(0)
        self.cam.snap()
        self.status("Scan Complete")

