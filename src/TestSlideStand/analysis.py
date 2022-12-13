import pathlib
import sys
from typing import Dict, Any, Optional

import png
import cv2
import warnings
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QApplication
from pydispatch import dispatcher

import matplotlib.pyplot as plt
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
)
from skimage.measure import find_contours
from skimage.filters import threshold_otsu

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
            area_pix = cv2.contourArea(cnt) / projection
            return approx, area_pix, cnt


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


def flatten_2darray(Map):
    num_row, num_col = Map.shape
    Cols, Rows = np.meshgrid(np.arange(num_col), np.arange(num_row))
    y = Rows.flatten()
    x = Cols.flatten()
    z = Map.flatten()
    return x, y, z


def clean_line_fit(x, y):
    p = np.polyfit(x, y, 1)
    y_fit = np.polyval(p, x)
    residual = y - y_fit
    sigma = np.std(residual)
    keep = np.where(np.abs(residual) < 3 * sigma)
    x_cln = x[keep]
    y_cln = y[keep]
    return np.polyfit(x_cln, y_cln, 1)


def analyze_spatial_intensity(
        image,
        project_angle,
        down_sample_ratio,
        boundary_length_threshold=4000,
        width_clearance=10,
        width_bright_edge=100
):
    image = downsample_image(image, down_sample_ratio)
    th_adaptive = adaptiveThreshold(image, round_up_to_odd(1001 / down_sample_ratio))
    th_close = binary_closing(th_adaptive, disk(5))
    th = binary_opening(th_close, disk(5))
    th = th.astype(np.uint8)

    contours = find_contours(th, 0.5)
    rec_props = find_black_box(contours, project_angle, boundary_length_threshold / down_sample_ratio)

    if rec_props is None:
        warnings.warn('Box not detected!')
        intensity = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        return intensity, None
    else:
        approx_rec, area_rec, contour_rec = rec_props
        bbox_rec, mask_sign = reorder_bbox_rec(np.squeeze(approx_rec))

        bbox_int = list(bbox_rec + (width_clearance // down_sample_ratio) * mask_sign)
        bbox_ext = list(bbox_int + (width_bright_edge // down_sample_ratio) * mask_sign)

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
        return intensity, [down_sample_ratio * bbox_rec, area_rec * down_sample_ratio ** 2,
                           down_sample_ratio * contour_rec]


def calculate_edge_parallelism(image, rec_props, margin_pix=100):
    """
    Calculate the angle between black rectangle box and glass edge
    Parameters:
        image: 2D array
        rec_props: Properties of black rectangle as 2nd output of analyze_spatial_intensity.
        margin_pix: half width of the ROI mask
    Returns:
        angle_rec2edge: angle between rectangle and glass edge
        p_rec_bottom: linear fit result of rectangle bottom line
        p_edge: linear fit of glass bottom edge
    """
    rec_area_mm = 12 * 24  # area of black rectangle
    dis_edge_mm = 6.605  # distance of rectangle edge to glass edge
    filter_size = 3  # Size of smoothing filter
    bbox_rec, area_pix, contour_rec = rec_props
    cnt_rec_x, cnt_rec_y = contour_rec.T
    pix_size = (rec_area_mm / area_pix) ** 0.5
    dis_edge_pix = dis_edge_mm / pix_size

    bbox_rec_bottom = np.squeeze(
        np.array([bbox_rec[0, :], bbox_rec[1, :], bbox_rec[1, :], bbox_rec[0, :]]) + margin_pix * np.array(
            [[[-1, 1], [1, 1], [1, -1], [-1, -1]]]))

    xlim_rec_bottom = [bbox_rec_bottom[2, 0], bbox_rec_bottom[0, 0]]

    ylim_rec_bottom = [bbox_rec_bottom[2, 1], bbox_rec_bottom[0, 1]]

    idx_rec_bottom = np.logical_and.reduce((
        cnt_rec_x > xlim_rec_bottom[0],
        cnt_rec_x < xlim_rec_bottom[1],
        cnt_rec_y > ylim_rec_bottom[0],
        cnt_rec_y < ylim_rec_bottom[1]
    ))

    rec_bottom_point_x, rec_bottom_point_y = np.transpose(contour_rec[idx_rec_bottom, :])
    p_rec_bottom = clean_line_fit(rec_bottom_point_x, rec_bottom_point_y)
    angle_rec = 180 * np.arctan(p_rec_bottom[0]) / np.pi

    bbox_bottom_edge = bbox_rec_bottom.copy()
    bbox_bottom_edge[:, 1] = bbox_bottom_edge[:, 1] + dis_edge_pix
    mask_bottom_edge = make_rect_mask(bbox_bottom_edge, image.shape)

    kernel = np.ones((filter_size, filter_size), np.float32) / filter_size ** 2
    image_smooth = cv2.filter2D(image, -1, kernel)

    th_image = image_smooth > threshold_otsu(image_smooth)
    th_bottom_edge = th_image * mask_bottom_edge

    edge_point_x, edge_point_y, edge_point_z = flatten_2darray(th_bottom_edge)
    mask_edge_points = edge_point_z == 1
    edge_point_x = edge_point_x[mask_edge_points]
    edge_point_y = edge_point_y[mask_edge_points]

    p_edge = clean_line_fit(edge_point_x, edge_point_y)
    angle_edge = 180 * np.arctan(p_edge[0]) / np.pi
    angle_rec2edge = angle_rec - angle_edge
    return angle_rec2edge, p_rec_bottom, p_edge


## LIBS ABOVE

class ImageAnalyzer:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self._imgs: Dict[float, np.ndarray] = {}
        self._results: Dict[float, Dict[str, Any]] = {}
        self.df: Optional[pd.DataFrame] = None

    def reset(self):
        self._imgs = {}
        self._results = {}
        self.df = None

    def save(self, datadir: Path, ref: str):
        # Save images
        for angle, img in self._imgs.items():
            fname = datadir / f'{angle:.0f}deg.png'
            self.log.info("Writing %s", fname)
            with open(fname, 'wb') as f:
                w = png.Writer(width=img.shape[1], height=img.shape[0], greyscale=True, compression=False)
                w.write(f, img)

        # Save data
        self.df = pd.DataFrame(self._results.values(), index=self._results.keys()).sort_index()
        self.df.index.name = 'angle'
        self.df.to_csv(datadir/f'{ref}_data.csv')
        self.plot(self.df, datadir/f'{ref}.pdf', ref, display=False)
        self.plot(self.df, datadir/f'{ref}.png', ref, display=False)

    def add_image(self, angle: float, img):
        self._imgs[angle] = img
        self.process_image(img, angle)

    def process_image(self, img: np.ndarray, angle: float):
        self.log.info("Processing image for angle %.2f", angle)
        intensity, rec_props = analyze_spatial_intensity(img, angle, down_sample_ratio=DOWN_SAMPLE_FACTOR)

        parallelism = np.nan
        if angle == 0.0:
            try:
                parallelism, _, _ = calculate_edge_parallelism(img, rec_props)
                dispatcher.send('PARALLEL', dispatcher.Any, parallelism)
            except:
                self.log.warning("Parallelism failed?")
                pass

        self._results[angle] = {
            'Left': intensity[0],
            'Top left': intensity[1],
            'Top right': intensity[2],
            'Right': intensity[3],
            'Bottom right': intensity[4],
            'Bottom left': intensity[5],
            'parallelism': parallelism,
        }
        dispatcher.send('NEWDATA', dispatcher.Any)


    def load_offline(self, basepath: pathlib.Path):
        """
        Load from an input directory.
        Expects a series of files named [angle]deg.png
        :param basepath:
        :return:
        """

        pngs = basepath.glob('*deg.png')
        fnames = {}
        for img in pngs:
            try:
                fnames[int(img.stem.replace('deg', ''))] = img

            except ValueError:
                self.log.warning('Skipping file %s', img)
                pass

        if len(fnames) == 0:
            raise FileNotFoundError("Didn't find and valid pngs")

        # self._imgs = {}
        # self._intensity = {}
        for angle, fname in fnames.items():
            self.log.info("Loading angle %.2f / %s", angle, fname)
            with open(fname, 'rb') as f:
                img = png.Reader(f).read()
                img = np.vstack(list(map(np.uint8, img[2])))
                self.add_image(angle, img)

        self.df = pd.DataFrame(self._results.values(), index=self._results.keys()).sort_index()

    # TODO unify plotting and GUI analysis instead of duplciating below
    # TODO only create the df in one location

    def finalize(self):
        df = pd.DataFrame(self._results.values(), index=self._results.keys()).sort_index()
        df.index.name = 'angle'

        int_left = df[['Top left', 'Bottom left']]
        int_right = df[['Top right', 'Bottom right']]
        int_left_mean = np.nanmean(int_left, axis = 1)
        int_right_mean = np.nanmean(int_right, axis = 1)
        int_angle_left = 100*(int_left_mean[0] - int_left_mean[-1])/ int_left_mean.mean()
        int_angle_right = 100*(int_right_mean[0] - int_right_mean[-1]) / int_right_mean.mean()
        int_mean = (int_left_mean + int_right_mean)/2
        int_spatial_var = 100 * (int_left_mean - int_right_mean) / (12 * int_mean)

        angle_dep = np.max([int_angle_left, int_angle_right])
        spatial_dep = int_spatial_var.max()

        dispatcher.send('FINAL_ANGULAR', dispatcher.Any, angle_dep)
        dispatcher.send('FINAL_SPATIAL', dispatcher.Any, spatial_dep)

    def plot(self, df: pd.DataFrame, fname: pathlib.Path, ref: str, display=False):
        int_left = df[['Top left', 'Bottom left']]
        int_right = df[['Top right', 'Bottom right']]
        int_left_mean = np.nanmean(int_left, axis = 1)
        int_right_mean = np.nanmean(int_right, axis = 1)
        int_angle_left = 100*(int_left_mean[0] - int_left_mean[-1])/ int_left_mean.mean()
        int_angle_right = 100*(int_right_mean[0] - int_right_mean[-1]) / int_right_mean.mean()
        int_mean = (int_left_mean + int_right_mean)/2
        int_spatial_var = 100 * (int_left_mean - int_right_mean) / (12 * int_mean)
        parallelism = df['parallelism']

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        ax1: plt.Axes
        ax2: plt.Axes

        fig.suptitle('Plots for %s \n Parallelism = %.3f deg' %(ref, np.nanmean(parallelism)) )

        ax1.set_title('Light intensity - angle dependence \n max = %.1f%%' % np.max([int_angle_left, int_angle_right]))
        for col in ['Left', 'Top left', 'Bottom left', 'Top right', 'Right', 'Bottom right']:
            ax1.plot(df.index, df[col], '-o', label=col)

        ax1.legend(loc='best')
        ax1.set_xlabel('Angles (deg)')
        ax1.set_ylabel('Intensity')
        ax1.grid()

        ax2.set_title('Light intensity - spatial dependence \n max = %.1f%%' % int_spatial_var.max())
        ax2.plot(df.index, int_spatial_var, '-o')
        ax2.grid()
        ax2.set_xlabel('Angles (deg)')
        ax2.set_ylabel('Intensity variation (% /mm)')

        fig.tight_layout()
        fig.savefig(fname)
        if display:
            plt.ion()
            plt.show(block=True)
        return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

    while 1:
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)
        dialog = QFileDialog()
        dialog.FileMode = dialog.FileMode.Directory
        dialog.ViewMode = QFileDialog.ViewMode.Detail
        inpath = dialog.getExistingDirectory()
        if inpath:
            inpath = Path(inpath)
            a = ImageAnalyzer()
            a.load_offline(inpath)
            a.plot(df=a.df, fname=inpath / 'output.pdf', ref=str(inpath), display=True)
        else:
            break

