import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep
from mplhep import label as label_base
from tests import ragged_to_normal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def find_axis_label(label):
    if "x_pos" in label or "y_pos" in label:
        return "on_callback_histogram_pos_label"
    if "width" in label or "height" in label:
        return "on_callback_histogram_length_label"
    if "energy" in label:
        return "on_callback_histogram_energy_label"
    return ""


def prepare_dimension_columns(dim_min, steps, cell_size):
    dim_cols = []
    for i in range(steps):
        dim_cols.append(dim_min + i * cell_size)
    return dim_cols


def ensure_array(elem):
    if isinstance(elem, (list, np.ndarray)):
        return elem
    return [elem]


def plot_shapes(ax, ys, **kwargs):
    for min_x, max_x, min_y, max_y in zip(
        ensure_array(ragged_to_normal(ys[:, 0]).tolist()),
        ensure_array(ragged_to_normal(ys[:, 2]).tolist()),
        ensure_array(ragged_to_normal(ys[:, 1]).tolist()),
        ensure_array(ragged_to_normal(ys[:, 3]).tolist()),
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
    kwargs.setdefault("fontsize", 24)
    kwargs.setdefault("fontname", "Tex Gyre Termes")
    kwargs.setdefault("exp_weight", "bold")
    kwargs.setdefault("loc", 4)  # 4 top right, underneath the axis
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


def plot_energy_resolution(
    ax,
    data_tuple,
    data_labels,
    data_colors,
    min_energy=100.0,  # 100 MeV
    max_energy=1e5,  # 100 GeV
    bins=50,
    log_scale=False,
    **kwargs
):
    span = abs(max_energy - min_energy)
    if span <= 1.0:
        min_energy -= 10.0
        max_energy += 10.0

    if span > 1e3:
        log_scale = True

    energy_true = data_tuple[0]
    energy_pred = data_tuple[1]
    diff = energy_pred - energy_true
    bin_size = (max_energy - min_energy) / bins
    xbins = np.arange(min_energy, max_energy + bin_size, bin_size)
    if log_scale:
        xbins = np.logspace(np.log10(min_energy), np.log10(max_energy), num=bins)
    xbinst = np.array([[xbins[i], xbins[i + 1]] for i in range(len(xbins[:-1]))])
    ybins = np.array(
        [
            100.0
            * np.mean(np.abs(diff[(energy_true > x[0]) & (energy_true <= x[1])]))
            / np.mean(x)
            for x in xbinst
        ]
    )
    ybins_stddev = np.array(
        [
            100.0
            * np.std(np.abs(diff[(energy_true > x[0]) & (energy_true <= x[1])]))
            / np.mean(x)
            for x in xbinst
        ]
    )
    hep.histplot(ybins, xbins, histtype="errorbar", xerr=True, yerr=ybins_stddev, ax=ax)
    if log_scale:
        ax.set_xscale("log")
    plt.ylabel(r"$\frac{\sigma}{E}$ [$\%$]")
    plt.xlabel("Energy [MeV]")


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


def plot_confusion_matrix(
    ax,
    data_tuple,
    class_names,
):
    if len(data_tuple[0]) == len(data_tuple[1]) == 0:
        return
    cm = confusion_matrix(
        data_tuple[0], data_tuple[1], labels=list(range(len(class_names)))
    )
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax)
