import logging
from harvesters.core import Harvester
import numpy as np
import coloredlogs

__all__ = ['Cam']

class Cam:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info("Init")
        h = self.h = Harvester()
        h.add_file(r'C:\Program Files\FLIR Systems\Spinnaker\cti64\vs2015\FLIR_GenTL_v140.cti')
        h.update()
        if not len(h.device_info_list):
            raise RuntimeError("No camera found")
        cam = h.create({'display_name': 'FLIR Blackfly S BFS-U3-120S4M'})
        self.cam = cam
        self.nodemap = self.cam.remote_device.node_map

    def __del__(self):
        self.cam.destroy()
        self.h.reset()

    def snap(self) -> np.ndarray:
        shape = (self.nodemap.Height.value, self.nodemap.Width.value)
        self.cam.start()
        buf = self.cam.fetch()
        ret = buf.payload.components[0].data.copy().reshape(shape)
        buf.queue()
        self.cam.stop()
        return ret

    def setup(self):
        nodemap = self.nodemap
        self.log.info("Setting up camera")
        nodemap.AcquisitionMode.set_value('SingleFrame')
        nodemap.ExposureAuto.set_value('Off')
        nodemap.GainAuto.set_value('Off')
        nodemap.Gain.set_value(0)
        self.log.info("Setup complete")

    def auto_exposure(self, iter=100):
        """
        Turn on camera autoexposure. Expose 100 frames. Stop Camera.
        Take the resulting exposure settings as the autoexposure value.
        :return:
        """
        nodemap = self.nodemap
        self.log.info("Starting autoexposure")
        nodemap.AcquisitionMode.set_value('Continuous')
        nodemap.ExposureAuto.set_value('Continuous')
        self.cam.start()
        for i in range(iter):
            buf = self.cam.fetch()
            buf.queue()
            self.log.info("Iteration %d: Exposure: %.2f", i, nodemap.ExposureTime.get_value(ignore_cache=True))
        self.cam.stop()
        nodemap.ExposureAuto.set_value('Off')
        return nodemap.ExposureTime.value

    def exp(self, exposure: float):
        self.log.info("Setting camera exposure to %.2f", exposure)
        self.nodemap.ExposureTime.set_value(exposure)
        self.log.info("Actual exposure: %.3f", self.nodemap.ExposureTime.value)
        # self.cam.ExposureTime(int(exposure))

if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG)
    c = Cam()
    print(c.snap())
    print("Done")
