from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('QT5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplWidget(FigureCanvasQTAgg):
    labels = ('Left', 'Top left', 'Bottom left', 'Top right', 'Right', 'Bottom right')

    def __init__(self, parent=None, width=3, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super(MplWidget, self).__init__(self.fig)
        self.setParent(parent)
        self.axs = self.fig.subplots(nrows=2)
        self.axs: List[matplotlib.figure.Axes]
        self.angle_plots: Dict[str, plt.Line2D] = {}
        self.setup_plots()
        self.updateGeometry()

    def reset(self):
        ax1, ax2 = self.axs
        ax1.cla()
        ax2.cla()
        self.draw()

    def setup_plots(self):
        fig = self.fig
        ax1, ax2 = self.axs

        ax1.set_title('Light intensity - angle dependence')
        ax1.legend(loc='best')
        ax1.set_xlabel('Angles (deg)')
        ax1.set_ylabel('Intensity')
        ax1.grid()
        for label in self.labels:
            self.angle_plots[label] = ax1.plot([], [], '-o', label=label)[0]
        ax1.legend(loc='best')
        ax1.set_xlim(-30, 30)

        ax2.set_title('Light intensity - spatial dependence')
        self.spatial_plot = ax2.plot([], [], '-o')[0]
        ax2.grid()
        ax2.set_xlabel('Angles (deg)')
        ax2.set_ylabel('Intensity variation (%/mm)')
        ax2.set_xlim(-30, 30)

        # fig.tight_layout()

    def plot_data(self, df: pd.DataFrame):
        int_left = df[['Left', 'Top left', 'Bottom left']]
        int_right = df[['Top right', 'Right', 'Bottom right']]
        int_spatial_var = 100 * (np.nanmean(int_left, axis=1) - np.nanmean(int_right, axis=1)) / (12 * df.mean(axis=1))
        ax1, ax2 = self.axs

        self.reset()

        ax1.set_title('Light intensity - angle dependence')
        ax1.legend(loc='best')
        ax1.set_xlabel('Angles (deg)')
        ax1.set_ylabel('Intensity')
        ax1.grid()
        for label in self.labels:
            self.angle_plots[label] = ax1.plot(df.index, df[label], '-o', label=label, alpha=0.75)[0]
        ax1.legend(loc='best')
        ax1.set_xlim(-30, 30)
        ax1.set_ylim(0, 255)

        ax2.set_title('Light intensity - spatial dependence')
        self.spatial_plot = ax2.plot([], [], '-o')[0]
        ax2.grid()
        ax2.set_xlabel('Angles (deg)')
        ax2.set_ylabel('Intensity variation (%/mm)')
        ax2.set_xlim(-30, 30)
        ax2.set_ylim(0, 10.0)

        for label in self.labels:
            self.angle_plots[label] = ax1.plot(df.index, df[label], '-o', label=label, alpha=0.75)[0]

        self.spatial_plot = ax2.plot(df.index, int_spatial_var, '-o')[0]
        self.draw()
