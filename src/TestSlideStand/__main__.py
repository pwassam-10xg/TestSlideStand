import logging
from pathlib import Path
import coloredlogs

from TestSlideStand import TestSlideStand

def main():
    coloredlogs.install(level=logging.DEBUG)
    t = TestSlideStand(
        TestSlideStand.Settings(
            zaber_sn='AC01ZPBDA',
        )
    )
    data = t.scan()

    import png
    basepath = Path('data')
    basepath.mkdir(parents=True, exist_ok=True)

    for angle, img in data.items():
        fname = basepath/f'{angle:.0f}.png'
        logging.info("Writing %s", fname)
        with open(fname, 'wb') as f:
            w = png.Writer(width=img.shape[0], height=img.shape[1], greyscale=True)
            w.write(f, img.T.copy())
    return

if __name__ == '__main__':
    main()