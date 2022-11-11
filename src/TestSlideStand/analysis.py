import pathlib
from typing import Dict

import png
import cv2
import os
import warnings
import logging
import pandas as pd
import numpy as np

# import gif
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
)
from skimage.measure import find_contours

DOWN_SAMPLE_FACTOR = 2


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

def downsample_image(image, ratio, interp=cv2.INTER_LINEAR):
    image_ds = cv2.resize(image, (image.shape[1] // ratio, image.shape[0] // ratio), interpolation=interp)
    return image_ds

def adaptiveThreshold(image, blockSize):
    th = cv2.adaptiveThreshold(
        image,
        maxValue=1,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=blockSize,
        C=0
    )
    return th

def reorder_bbox_rec(bbox_rec):
    """
    Reorder rectangle bbox (or square) in the order quodrant 1, 2, 3 and 4
    """
    center_x, center_y = np.mean(bbox_rec, axis=0)
    mask_sign = (np.vstack(((bbox_rec[:, 0] - center_x) > 0, (bbox_rec[:, 1] - center_y) > 0)).T).astype(int)
    mask_sign[mask_sign == 0] = -1
    mask_sign_x, mask_sign_y = mask_sign.T
    idx_tl = np.logical_and(mask_sign_x == -1, mask_sign_y == -1)
    idx_tr = np.logical_and(mask_sign_x == 1, mask_sign_y == -1)
    idx_br = np.logical_and(mask_sign_x == 1, mask_sign_y == 1)
    idx_bl = np.logical_and(mask_sign_x == -1, mask_sign_y == 1)
    bbox_reorder = np.squeeze(
        np.array([bbox_rec[idx_br, :], bbox_rec[idx_bl, :], bbox_rec[idx_tl, :], bbox_rec[idx_tr, :]]))
    mask_sign = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    return bbox_reorder, mask_sign


def make_rect_mask(bbox, size):
    num_row, num_col = size
    blank_array = np.zeros((num_row, num_col))
    bbox_reorder, _ = reorder_bbox_rec(bbox)
    mask = cv2.fillPoly(blank_array, [bbox_reorder.reshape(-1, 1, 2)], 1)
    return mask.astype('float')


def func_fit2dPoly(x, y, z, order):
    '''
    Inputs:
    x, y, z are 1d np.array specifying the points to be fitted. The three vectors must be the same length.
    Order is the order of the polynomial to fit.

    Return:
    Coefficients of the polynomial.  These are in increasing power of y for each increasing power of x, e.g. for order 2:
    zbar = coeffs(1) + coeffs(2).*y + coeffs(3).*y^2 + coeffs(4).*x + coeffs(5).*x.*y + coeffs(6).*x^2
    '''

    if len(y) != len(x) or len(z) != len(x):
        raise Exception("Inputs vectors must be the same length")

    numVals = len(x)

    # number of combinations of coefficients in resulting polynomial
    numCoeffs = int((order + 2) * (order + 1) / 2)

    # Form array to process with SVD
    A = np.zeros((numVals, numCoeffs))

    column = 0
    for xpower in range(order + 1):
        for ypower in range(order - xpower + 1):
            A[:, column] = (x ** xpower) * (y ** ypower)
            column += 1

    polyCoeff = np.matmul(np.linalg.pinv(A), z)
    zFit = np.matmul(A, polyCoeff)
    return polyCoeff, zFit


def func_fit2dPoly_map(map, order):
    '''
    map: image to fit
    order: order of the polynomial to fit.

    Return:
    Coefficients of the polynomial.  These are in increasing power of y for each increasing power of x, e.g. for order 2:
    zbar = coeffs(1) + coeffs(2).*y + coeffs(3).*y^2 + coeffs(4).*x + coeffs(5).*x.*y + coeffs(6).*x^2
    '''
    nRow = len(map)
    nCol = len(map[0])
    Cols, Rows = np.meshgrid(list(range(nCol)), list(range(nRow)), copy=False)
    x = Cols.flatten()
    y = Rows.flatten()
    z = map.flatten()
    mask_notnan = ~np.isnan(z)
    x = x[mask_notnan]
    y = y[mask_notnan]
    z = z[mask_notnan]

    polyCoeff, _ = func_fit2dPoly(x, y, z, order)

    return polyCoeff


def find_black_box(contours, project_angle, length_threshold):
    projection = np.cos(np.pi * project_angle / 180)
    for cnt in contours:
        cnt = np.fliplr(cnt.astype(np.int32))
        cnt_x, cnt_y = cnt.T
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = (float(w) / h) / projection
        if len(cnt_x) > length_threshold and len(approx) == 4 and ratio > 1.7 and ratio < 2.3:
            return approx


def analyze_spatial_intensity(
        image,
        project_angle,
        down_sample_ratio,
        boundary_length_threshold=4000,
        width_clearance=10,
        width_bright_edge=100
) -> np.ndarray:
    image = downsample_image(image, down_sample_ratio)
    th_adaptive = adaptiveThreshold(image, round_up_to_odd(1001 / down_sample_ratio))
    th_close = binary_closing(th_adaptive, disk(5))
    th = binary_opening(th_close, disk(5))
    th = th.astype(np.uint8)

    contours = find_contours(th, 0.5)
    approx_rec = find_black_box(contours, project_angle, boundary_length_threshold / down_sample_ratio)

    if approx_rec is None:
        warnings.warn('Box not detected!')
        intensity = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        bbox_rec, mask_sign = reorder_bbox_rec(np.squeeze(approx_rec))

        bbox_int = list(bbox_rec + (width_clearance // down_sample_ratio) * mask_sign)
        bbox_ext = list(bbox_int + (width_bright_edge // down_sample_ratio) * mask_sign)
        mask_ring = (make_rect_mask(np.array(bbox_ext), image.shape) - make_rect_mask(np.array(bbox_int),
                                                                                      image.shape)) * th

        bbox_left = [bbox_int[1], bbox_ext[1], bbox_ext[2], bbox_int[2]]
        bbox_top_left = [(bbox_int[3] + bbox_int[2]) // 2, bbox_int[2], bbox_ext[2], (bbox_ext[3] + bbox_ext[2]) // 2]
        bbox_top_right = [bbox_int[3], (bbox_int[3] + bbox_int[2]) // 2, (bbox_ext[3] + bbox_ext[2]) // 2, bbox_ext[3]]
        bbox_right = [bbox_ext[0], bbox_int[0], bbox_int[3], bbox_ext[3]]
        bbox_bottom_right = [bbox_ext[0], (bbox_ext[0] + bbox_ext[1]) // 2, (bbox_int[0] + bbox_int[1]) // 2,
                             bbox_int[0]]
        bbox_bottom_left = [(bbox_ext[0] + bbox_ext[1]) // 2, bbox_ext[1], bbox_int[1],
                            (bbox_int[0] + bbox_int[1]) // 2]

        bboxes = [bbox_left, bbox_top_left, bbox_top_right, bbox_right, bbox_bottom_right, bbox_bottom_left]

        intensity = []
        for bbox in bboxes:
            mask_bbox = make_rect_mask(np.array(bbox), image.shape) * th
            mask_bbox[mask_bbox == 0] = np.nan
            image_ROI = mask_bbox * image
            intensity_project = np.nanmean(image_ROI)
            # intensity.append(intensity_project * projection)
            intensity.append(intensity_project)
    return intensity


## LIBS ABOVE

class ImageAnalyzer:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self._imgs: Dict[float, np.ndarray] = {}
        self._intensity: Dict[float, np.ndarray] = {}

    def add_image(self, angle: float, img):
        self._imgs[angle] = img
        # TODO background process image here

    def process_image(self, img: np.ndarray, angle: float):
        intensity = analyze_spatial_intensity(img, project_angle=angle, down_sample_ratio=DOWN_SAMPLE_FACTOR)
        self._intensity[angle] = intensity

    def load_offline(self, basepath: pathlib.Path):
        """
        Load from an input directory.
        Expects a series of files named [angle].tiff
        :param basepath:
        :return:
        """

        tiffs = basepath.glob('*.tiff')
        fnames = {}
        for t in tiffs:
            try:
                fnames[int(t.stem)] = t

            except ValueError:
                self.log.warning('Skipping file %s', t)
                pass

        if len(fnames) == 0:
            raise FileNotFoundError("Didn't find and valid tiffs")

        self._imgs = {}
        self._intensity = {}
        for angle, fname in fnames.items():
            self.log.info("Loading angle %.2f / %s", angle, fname)
            self._imgs[angle] = png.Reader(fname).read()


    def plot(self, prefix: pathlib.Path):
        # Finish any pending plots

        prefix = pathlib.Path(prefix)
        df = pd.DataFrame(self._intensity.values(),
                          index=self._intensity.keys(),
                          columns=['Left', 'Top left', 'Top right', 'Right', 'Bottom right', 'Bottom left'])
        df = df.reset_index()

        df.to_csv(prefix.with_suffix('.csv'))

        int_left = df[['Left', 'Top left', 'Bottom Left']]
        int_right = df[['Top right', 'Right', 'Bottom right']]
        int_spatial_var = 100*(np.nanmean(int_left, axis = 1) - np.nanmean(int_right, axis = 1))/(12*np.nanmean(df.index, axis = 1))

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 5))
        ax1: plt.Axes
        ax2: plt.Axes

        plt.title('Light intensity - angle dependence')

        for col in df.columns:
            ax1.plot(df.index, df[col], '*', label=col)

        ax1.legend(loc='best')
        ax1.xlabel('Angles (deg)')
        ax1.ylabel('Intensity')
        ax1.grid()

        ax2.title('Light intensity - spatial dependence')
        ax2.plot(df.index, int_spatial_var, '*')
        ax2.grid()
        ax2.xlabel('Angles (deg)')
        ax2.ylabel('Intensity variation (% /mm)')

        fig.show()
        fig.savefig(prefix.with_suffix('.pdf'))
        return

if __name__ == '__main__':
    a = ImageAnalyzer()
    a.load_offline(pathlib.Path('testdata/'))
    a.plot(pathlib.Path('testdata/output.pdf'))

