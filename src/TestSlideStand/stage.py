import logging
import coloredlogs
from zaber_motion import Units
from zaber_motion.ascii import Connection
import serial.tools.list_ports

__all__ = ['Stage']

class Stage:
    def __init__(self, serno: str):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info("Init")

        # Use pyserial to identify the comport
        p = list(serial.tools.list_ports.grep(serno))
        if len(p) == 0:
            raise RuntimeError("Cant locate com port with serial number: %s", serno)
        if len(p) > 1:
            raise RuntimeError("Multiple matching serial ports?")
        port = p[0].name

        self.zab = Connection.open_serial_port(port)
        devices = self.zab.detect_devices()
        if len(devices) != 1:
            raise RuntimeError("Expected 1 device, got %d", len(devices))

        self.axis = devices[0].get_axis(1)
        return

    def home(self):
        self.log.info("Homing")
        if self.axis.is_homed():
            self.log.info("Already homed moving to abs 0")
            self.axis.move_absolute(0)
        self.axis.move_relative(45)
        self.axis.home()

    def abs(self, degrees: float):
        self.log.info("Moving absolute %.3f degrees", degrees)
        self.axis.move_absolute(degrees, Units.ANGLE_DEGREES)
        self.log.info("Move complete")

    def rel(self, degrees: float):
        self.log.info("Moving relative %.3f degrees", degrees)
        self.axis.move_relative(degrees, Units.ANGLE_DEGREES)
        self.log.info("Move complete")

if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG)
    s = Stage('AC01ZPBDA')
    s.home()


