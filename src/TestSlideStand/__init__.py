__version__ = '0.0.1'

import logging
import numpy as np
import png

from typing import Dict
from dataclasses import dataclass
from pathlib import Path

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

    def shutdown(self):
        self.zaber.shutdown()
        self.cam.shutdown()
        return

    def scan(self) -> Dict[float, np.ndarray]:
        self.log.info("Scanning")

        self.log.info("Auto exposure")
        self.zaber.abs(self.settings.auto_exp_pos)
        exp = self.cam.auto_exposure()
        self.log.info("Auto exposure at %.1f uS", exp)

        for p in self.settings.positions:
            self.log.info("Scanning position %.1f", p)
            self.zaber.abs(p)
            img = self.cam.snap()
            self.data[p] = img

        self.log.info("Scan Complete")

    def save(self, datadir: Path):
        for angle, img in self.data.items():
            fname = datadir / f'{angle:.0f}deg.png'
            self.log.info("Writing %s", fname)
            with open(fname, 'wb') as f:
                w = png.Writer(width=img.shape[1], height=img.shape[0], greyscale=True, compression=False)
                w.write(f, img)
