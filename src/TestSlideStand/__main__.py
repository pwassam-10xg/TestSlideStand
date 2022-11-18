import logging
import sys

from PyQt5.QtWidgets import QApplication

from TestSlideStand.GUI import TestSlideStandGUI
from TestSlideStand.settings import settings

def main():
    logging.basicConfig(format='%(asctime)s %(name)20s %(levelname)s %(message)s', level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

    # Make sure data directory exists
    settings.datadir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    win = TestSlideStandGUI()
    app.exec()

if __name__ == '__main__':
    main()