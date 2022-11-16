from typing import List

import matplotlib
matplotlib.use('QT5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplWidget(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=3, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)

        self.axs = self.fig.subplots(nrows=2)
        self.axs: List[matplotlib.figure.Axes]

        self.setup_plots()

    def setup_plots(self):
        fig = self.fig
        ax1, ax2 = self.axs

        ax1.set_title('Light intensity - angle dependence')
        ax1.legend(loc='best')
        ax1.set_xlabel('Angles (deg)')
        ax1.set_ylabel('Intensity')
        ax1.grid()
        self.plot_angle = ax1.plot([], [], '*')

        ax2.set_title('Light intensity - spatial dependence')
        self.plot_spatial = ax2.plot([], [], '*')
        ax2.grid()
        ax2.set_xlabel('Angles (deg)')
        ax2.set_ylabel('Intensity variation (% /mm)')

        fig.tight_layout()
