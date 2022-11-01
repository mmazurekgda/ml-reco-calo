import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep
from mplhep import label as label_base
from tests import ragged_to_normal


def find_axis_label(label):
    if "x_pos" in label or "y_pos" in label:
        return "on_epoch_histogram_pos_label"
    if "width" in label or "height" in label:
        return "on_epoch_histogram_length_label"
    if "energy" in label:
        return "on_epoch_histogram_energy_label"
    return ""


def prepare_dimension_columns(dim_min, steps, cell_size):
    dim_cols = []
    for i in range(steps):
        dim_cols.append(dim_min + i * cell_size)
    return dim_cols


def plot_shapes(ax, ys, **kwargs):
    for min_x, max_x, min_y, max_y in zip(
        ragged_to_normal(ys[:, 0]).tolist(),
        ragged_to_normal(ys[:, 2]).tolist(),
        ragged_to_normal(ys[:, 1]).tolist(),
        ragged_to_normal(ys[:, 3]).tolist(),
    ):
        ax.add_artist(
            plt.Rectangle(
                (min_x, min_y),
                height=(max_y - min_y),
                width=(max_x - min_x),
                fill=False,
                **kwargs
            )
        )


def add_lhcb_like_label(label=None, exp="LHCb", **kwargs):
    for key, value in dict(hep.rcParams.label._get_kwargs()).items():
        if (
            value is not None
            and key not in kwargs.keys()
            and key in inspect.getfullargspec(label_base.exp_label).kwonlyargs
        ):
            kwargs.setdefault(key, value)
    kwargs.setdefault("italic", (False, False))
    kwargs.setdefault("fontsize", 28)
    # kwargs.setdefault("fontname", "Times New Roman")
    kwargs.setdefault("exp_weight", "normal")
    kwargs.setdefault("loc", 4)  # 4 top right, underneath the axis
    kwargs.setdefault("exp_weight", "normal")
    if label is not None:
        kwargs["label"] = label
    return label_base.exp_label(exp, **kwargs)


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


def plot_event(
    ax,
    data_tuple,
    data_labels,
    data_colors,
    min_hit_energy=0.0,
    img_x_min=0.0,
    img_x_max=1.0,
    img_y_min=0.0,
    img_y_max=1.0,
    img_width=1.0,
    img_height=1.0,
    **kwargs
):
    energies, ys, preds = data_tuple[0], data_tuple[1], data_tuple[2]
    energies[energies <= min_hit_energy] = np.NaN
    cell_x = (img_x_max - img_x_min) / img_width
    cell_y = (img_y_max - img_y_min) / img_height
    x_cols = prepare_dimension_columns(img_x_min + cell_x / 2, img_width, cell_x)
    y_cols = prepare_dimension_columns(img_y_min + cell_y / 2, img_height, cell_y)
    im = ax.pcolormesh(x_cols, y_cols, energies, **kwargs)
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
    cbar.set_label("Energy Deposited [MeV]")
    plot_shapes(ax, ys, color="blue", hatch="/")
    plot_shapes(ax, preds, color="red", hatch="\\")
