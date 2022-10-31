import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep


def find_axis_label(label):
    if "x_pos" in label or "y_pos" in label:
        return "on_epoch_histogram_pos_label"
    if "width" in label or "height" in label:
        return "on_epoch_histogram_length_label"
    if "energy" in label:
        return "on_epoch_histogram_energy_label"
    return ""


def plot_histograms(
    ax,
    data_tuple,
    data_labels,
    data_colors,
    bins=100,
    offset_bins_no=3,
    histtype="step",
    **kwargs
):
    plot_bins = bins
    merged = np.concatenate(data_tuple, -1)
    min_value, max_value = merged.min(), merged.max()
    span = abs(max_value - min_value)
    step_width = span / bins

    try:
        plot_bins = np.arange(
            min_value - offset_bins_no * step_width,
            max_value + offset_bins_no * step_width,
            step_width,
        )
    except ValueError:
        plot_bins = bins
        pass

    for data, data_label, data_color in zip(data_tuple, data_labels, data_colors):
        hist, bins = np.histogram(data, bins=plot_bins)
        hep.histplot(
            hist,
            bins,
            ax=ax,
            histtype="errorbar",
            xerr=True,
            label=data_label,
            color=data_color,
        )


def plot_scatter_plots(ax, data_tuple, data_labels, data_colors, **kwargs):
    for data, data_label, data_color in zip(data_tuple, data_labels, data_colors):
        ax.scatter(
            data[0],
            data[1],
            label=data_label,
            color=data_color,
        )
