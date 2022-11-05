import logging
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

    # import png
    # for angle, img in data.items():
    #     with open(f'{angle:.0f}.png', 'wb') as f:
    #         w = png.Writer(width=img.shape[0], height=img.shape[1], greyscale=True)
    #         w.write(f, img.T.copy())

if __name__ == '__main__':
    main()