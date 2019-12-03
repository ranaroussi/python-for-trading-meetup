#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as _mpl
import matplotlib.pyplot as _plt
from mpl_toolkits.mplot3d import Axes3D as _Axes3D
import numpy as _np
import pandas as _pd


def cagr(returns):
    total = returns.add(1).prod() - 1
    years = len(set(returns.index.year))
    res = abs(total / 1.0) ** (1.0 / years) - 1

    if isinstance(returns, _pd.DataFrame):
        res = _pd.Series(res)
        res.index = returns.columns
    return res


def compsum(returns):
    return returns.add(1).cumprod() - 1


def max_drawdown(returns):
    prices = 1 + 1 * compsum(returns)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def sharpe(returns):
    return returns.mean() / returns.std() * _np.sqrt(252)


def optimize2d(data, run_strategy, param1, param2,
               param1_name="param1", param2_name="param2"):

    sharpes = _np.zeros((len(param1), len(param2)))
    drawdowns = _np.zeros(sharpes.shape)
    cagrs = _np.zeros(sharpes.shape)
    stds = _np.zeros(sharpes.shape)

    for i1, param1_val in enumerate(param1):
        for i2, param2_val in enumerate(param2):
            pnl = run_strategy(data, param1_val, param2_val)
            sharpes[i1, i2] = sharpe(pnl)
            drawdowns[i1, i2] = max_drawdown(pnl)
            cagrs[i1, i2] = cagr(pnl)
            stds[i1, i2] = pnl.std()

    return _optimizer(param1, param2,
                      param1_name, param2_name,
                      sharpes, drawdowns, cagrs, stds)


class _optimizer(object):

    def __init__(self, param1, param2,
                 param1_name, param2_name,
                 sharpes, drawdowns, cagrs, stds):

        self.param1 = param1
        self.param1_name = param1_name

        self.param2 = param2
        self.param2_name = param2_name

        self.sharpes = sharpes
        self.drawdowns = drawdowns
        self.cagrs = cagrs
        self.stds = stds

    def reveal(self):
        dd_param1, dd_param2 = _np.unravel_index(self.drawdowns.argmax(),
                                                 self.drawdowns.shape)

        shrp_param1, shrp_param2 = _np.unravel_index(self.sharpes.argmax(),
                                                     self.sharpes.shape)

        cagr_param1, cagr_param2 = _np.unravel_index(self.cagrs.argmax(),
                                                     self.cagrs.shape)

        std_param1, std_param2 = _np.unravel_index(self.stds.argmax(),
                                                   self.stds.shape)

        return {
            "sharpe": {
                self.param1_name: self.param1[shrp_param1],
                self.param2_name: self.param2[shrp_param2]
            },
            "drawdown": {
                self.param1_name: self.param1[dd_param1],
                self.param2_name: self.param2[dd_param2]
            },
            "cagr": {
                self.param1_name: self.param1[cagr_param1],
                self.param2_name: self.param2[cagr_param2]
            },
            "volatility": {
                self.param1_name: self.param1[std_param1],
                self.param2_name: self.param2[std_param2]
            }
        }

    def plot(self, figsize=(10, 8), show=True):
        fig, ((ax1, ax2), (ax3, ax4)) = _plt.subplots(2, 2, figsize=figsize)

        im1 = ax1.pcolormesh(self.param1, self.param2,
                             self.sharpes.T, cmap="jet")
        ax1.set_title("Sharpe", fontweight="bold", color="black")
        ax1.set_xlabel(self.param1_name)
        ax1.set_ylabel(self.param2_name)
        fig.colorbar(im1, ax=ax1, format=_plt.FormatStrFormatter('%.2f'))

        im2 = ax2.pcolormesh(self.param1, self.param2,
                             (self.drawdowns*100).T, cmap="jet")
        ax2.set_title("Drawdown", fontweight="bold", color="black")
        ax2.set_xlabel(self.param1_name)
        ax2.set_ylabel(self.param2_name)
        fig.colorbar(im2, ax=ax2, format=_plt.FormatStrFormatter('%.f%%'))

        im3 = ax3.pcolormesh(self.param1, self.param2,
                             (self.cagrs.T*100), cmap="jet")
        ax3.set_title("CAGR", fontweight="bold", color="black")
        ax3.set_xlabel(self.param1_name)
        ax3.set_ylabel(self.param2_name)
        fig.colorbar(im3, ax=ax3, format=_plt.FormatStrFormatter('%.f%%'))

        im4 = ax4.pcolormesh(self.param1, self.param2, self.stds.T, cmap="jet")
        ax4.set_title("Volatility", fontweight="bold", color="black")
        ax4.set_xlabel(self.param1_name)
        ax4.set_ylabel(self.param2_name)
        fig.colorbar(im4, ax=ax4, format=_plt.FormatStrFormatter('%.2f%%'))

        try:
            _plt.subplots_adjust(hspace=0)
        except Exception:
            pass
        try:
            fig.tight_layout()
        except Exception:
            pass

        if show:
            _plt.show()

        _plt.close(fig)

        if not show:
            return fig

    def plot3d(self, figsize=(8, 6)):

        # the fourth dimention is color
        color_dimension = (self.stds * 100)  # the fourth dimension
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = _mpl.colors.Normalize(minn, maxx)
        m = _plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

        fig = _plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')

        _plt.title("Optiimzation 4D Plot (color = 4th dimention)\n",
                   fontsize=14, fontweight="bold", color="black")
        surface = ax.plot_surface(self.sharpes, self.drawdowns*100, self.cagrs*100,
                                  linewidth=0., antialiased=True,
                                  facecolors=fcolors, vmin=minn, vmax=maxx,
                                  rstride=1, cstride=1)

        cbar = _plt.colorbar(m, shrink=0.7)
        cbar.ax.yaxis.set_major_formatter(_plt.FormatStrFormatter(' %.2f '))
        cbar.set_label("Volatility", fontweight="bold", color="black")

        ax.grid(True)
        ax.xaxis.pane.set_edgecolor('silver')
        ax.yaxis.pane.set_edgecolor('silver')
        # ax.zaxis.pane.set_edgecolor('silver')

        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False

        ax.xaxis.set_major_formatter(_plt.FormatStrFormatter(' %.2f '))
        ax.set_xlabel('\nSharpe', fontweight="bold", color="black")
        ax.yaxis.set_major_formatter(_plt.FormatStrFormatter(' %.f%% '))
        ax.set_ylabel('\n\nDrawdown', fontweight="bold", color="black")
        ax.zaxis.set_major_formatter(_plt.FormatStrFormatter('  %.f%% '))
        ax.set_zlabel('\nCAGR', fontweight="bold", color="black")

        try:
            _plt.subplots_adjust(hspace=0)
        except Exception:
            pass
        try:
            fig.tight_layout()
        except Exception:
            pass

        _plt.show()
