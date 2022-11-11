from dataclasses import dataclass
import logging
from typing import Dict

import coloredlogs

from camera import Cam
from stage import Stage
import numpy as np
from TestSlideStand.analysis import ImageAnalyzer

__all__ = ['TestSlideStand']

class TestSlideStand:
    @dataclass
    class Settings:
        zaber_sn: str                                    # FTDI serial number of zaber
        positions: np.ndarray = np.arange(-30, 30, 2)    # Positions to scan
        auto_exp_pos: float = -30                        # Auto exposure position in degrees

    def __init__(self, settings: Settings):
        self.log = logging.getLogger(self.__class__.__name__)
        self.settings = settings
        self.log.info("Init")

        self.zaber = Stage(settings.zaber_sn)
        self.cam = Cam()
        self.analyzer = ImageAnalyzer()

    def scan(self) -> Dict[float, np.ndarray]:
        self.log.info("Scanning")

        self.log.info("Auto exposure")
        self.zaber.abs(self.settings.auto_exp_pos)
        exp = self.cam.auto_exposure()
        self.log.info("Auto exposure at %.1f uS", exp)

        data = {}
        for p in self.settings.positions:
            self.log.info("Scanning position %.1f", p)
            self.zaber.abs(p)
            img = self.cam.snap()
            self.analyzer.add_image(p, img)


        self.log.info("Scan Complete")
        return data

if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG)
    t = TestSlideStand(
        TestSlideStand.Settings(
            zaber_sn='AC01ZPBDA',
            positions=[-10, 0, 10]
        )
    )
    data = t.scan()

    import png
    for angle, img in data.items():
        with open(f'{angle:.0f}.png', 'wb') as f:
            w = png.Writer(width=img.shape[0], height=img.shape[1], greyscale=True)
            w.write(f, img.T.copy())
