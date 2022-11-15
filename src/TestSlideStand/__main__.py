import logging
import time
import tkinter as tk
from tkinter import simpledialog
import TestSlideStand
from TestSlideStand.analysis import ImageAnalyzer
from TestSlideStand.settings import settings

def main():
    logging.basicConfig(format='%(asctime)s %(name)20s %(levelname)s %(message)s', level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

    # Make sure data directory exists
    settings.datadir.mkdir(parents=True, exist_ok=True)

    while 1:
        # Get slide ID
        ROOT = tk.Tk()
        ROOT.withdraw()
        ref = simpledialog.askstring(title="User input", prompt="slide ID: ")

        # Exit on empty slideid or dialog close
        if not ref or not ref.strip():
            break

        fname = settings.datadir / (ref + f'_{int(time.time())}')

        if fname.exists():
            raise FileExistsError

        fname.mkdir(exist_ok=False)

        # Fork logging system into the directory
        fh = logging.FileHandler(fname/'log.txt')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s %(name)20s %(levelname)s %(message)s'))
        logging.getLogger().addHandler(fh)
        logging.info("Log start")
        logging.info("TestSlideStand Version: %s", TestSlideStand.__version__)

        t = TestSlideStand.TestSlideStand(settings, ref)
        t.scan()
        t.save(fname)
        t.shutdown()
        del t

        a = ImageAnalyzer(ref)
        a.load_offline(fname)
        a.plot(fname)

        logging.info("Log end")
        logging.getLogger().removeHandler(fh)
        fh.close()

if __name__ == '__main__':
    main()