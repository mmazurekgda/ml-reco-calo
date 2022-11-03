import numpy as np
import tensorflow as tf
from tests import (
    prepare_dataset_for_inference,
    convert_data,
    ragged_to_normal,
)
import io
from matplotlib import pyplot as plt
import mplhep as hep
from collections import defaultdict

from vis import (
    plot_histograms,
    plot_scatter_plots,
    plot_event,
    plot_confusion_matrix,
    plot_energy_resolution,
    find_axis_label,
    add_lhcb_like_label,
)

registered_interruptions = 0


class StopTrainingSignal(Exception):
    pass


class CNNLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger=None):
        super().__init__()
        self.log = logger

    def stop_training_handler(self):
        def handler(signum, frame):
            global registered_interruptions
            registered_interruptions += 1
            if registered_interruptions == 1:
                self.log.warning(
                    "\nRegistered CTRL+C. Press again to stop the training safely."
                )
            elif registered_interruptions == 2:
                self.log.error(
                    "\nRegistered CTRL+C. Safe stopping. Press again to stop the training IMMEDIATELY."
                )
            else:
                self.log.fatal("\nStopping the training immediately!")
                raise StopTrainingSignal()

        return handler

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", "UNKNOWN")
        val_loss = logs.get("val_loss", "UNKNOWN")
        self.log.info(f"\n--> Epoch: {epoch}, 'loss': {loss}, 'val_loss': {val_loss}")

    def on_batch_end(self, batch, logs=None):
        global registered_interruptions
        if registered_interruptions >= 2:
            self.model.stop_training = True


class CNNTestingCallback(tf.keras.callbacks.Callback):
    non_automatic_histo_types = [
        "images",
        "true_position",
        "pred_position",
        "true_classes",
        "pred_classes",
        "matched_true_classes",
        "matched_pred_classes",
        "matched_true_energy",
        "matched_pred_energy",
        "ghost_x_pos",
        "missed_x_pos",
        "ghost_y_pos",
        "missed_y_pos",
        "ghost_x_energy",
        "missed_x_energy",
    ]

    joint_histograms = {
        "vs_particle_energy": {
            "histo_keys": ["true_energy", "pred_energy"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "vs_cluster_width": {
            "histo_keys": ["true_width", "pred_width"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "vs_cluster_height": {
            "histo_keys": ["true_height", "pred_height"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "vs_cluster_x_pos": {
            "histo_keys": ["true_x_pos", "pred_x_pos"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "vs_cluster_x_pos_extended": {
            "histo_keys": [
                "true_x_pos",
                "pred_x_pos",
                "matched_true_x_pos",
                "matched_pred_x_pos",
            ],
            "data_labels": [
                "Truth",
                "Predicted",
                "Matched Truth",
                "Matched Predicted",
            ],
            "data_colors": [
                "blue",
                "red",
                "cyan",
                "orange",
            ],
        },
        "vs_cluster_y_pos": {
            "histo_keys": ["true_y_pos", "pred_y_pos"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "vs_cluster_y_pos_extended": {
            "histo_keys": [
                "true_y_pos",
                "pred_y_pos",
                "matched_true_y_pos",
                "matched_pred_y_pos",
            ],
            "data_labels": [
                "Truth",
                "Predicted",
                "Matched Truth",
                "Matched Predicted",
            ],
            "data_colors": [
                "blue",
                "red",
                "cyan",
                "orange",
            ],
        },
    }

    def __init__(self, config, logger, dataset, image_transformation):
        super().__init__()
        self.log = logger
        self.config = config
        self.dataset = dataset
        self.image_transformation = image_transformation
        self.histogram_writer = tf.summary.create_file_writer(
            f"{self.config.tensorboard_log_dir}/testing_during_training"
        )
        self.timings = defaultdict(list)
        self.clusters_no = defaultdict(list)

    def _make_image_from_plot(
        self,
        name,
        data_tuple,
        data_labels=[],
        data_colors=[],
        xlabel=None,
        ylabel=None,
        plot_type="hist",
        exp=None,
        llabel=None,
        rlabel=None,
        loc=4,
        fontsize=20,
        title=None,
    ):
        if title is None:
            title = name
        if not exp:
            exp = self.config.testing_image_label_exp
        if not llabel:
            llabel = self.config.testing_image_label_llabel
        if not rlabel:
            rlabel = self.config.testing_image_label_rlabel
        plt.style.use(hep.style.LHCb2)
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(
                self.config.testing_image_figure_x_size,
                self.config.testing_image_figure_y_size,
            ),
        )
        add_lhcb_like_label(
            ax=ax,
            exp=exp,
            llabel=llabel,
            rlabel=rlabel,
            loc=loc,
            fontsize=fontsize,
        )
        if plot_type == "hist":
            plot_histograms(
                ax,
                data_tuple,
                data_labels,
                data_colors,
                bins=self.config.testing_image_histogram_buckets,
            )
        elif plot_type == "energy_resolution":
            plot_energy_resolution(
                ax,
                data_tuple,
                data_labels,
                data_colors,
                bins=self.config.testing_image_histogram_buckets,
                min_energy=self.config.min_particle_energy,
                max_energy=self.config.max_particle_energy,
            )
        elif plot_type == "scatter":
            plot_scatter_plots(
                ax,
                data_tuple,
                data_labels,
                data_colors,
            )
        elif plot_type == "event":
            plot_event(
                ax,
                data_tuple,
                data_labels,
                data_colors,
                min_hit_energy=self.config.min_hit_energy,
                img_x_min=self.config.img_x_min,
                img_x_max=self.config.img_x_max,
                img_y_min=self.config.img_y_min,
                img_y_max=self.config.img_y_max,
                img_width=self.config.img_width,
                img_height=self.config.img_height,
            )
        elif plot_type == "confusion_matrix":
            plot_confusion_matrix(
                ax,
                data_tuple,
                self.config.classes,
            )
            # aspect ratio here is different... use x
            fig.set_size_inches(
                self.config.testing_image_figure_x_size,
                self.config.testing_image_figure_x_size,
            )
        else:
            raise NotImplementedError()

        if title:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if plot_type in ["hist", "scatter"]:
            hep.plot.hist_legend(ax=ax, loc=1)
            # hep.plot.mpl_magic(ax=ax)
            hep.plot.ylow(ax=ax)
            hep.plot.yscale_legend(ax=ax)
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[-1] * 1.15)
            fig.canvas.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def _make_epoch_histograms(self, step, histo_data):
        self.log.debug("-> Generating histograms...")
        with self.histogram_writer.as_default():
            for histo_type, histo_values in histo_data.items():
                name = getattr(self.config, f"on_epoch_histogram_{histo_type}_name")
                group = getattr(self.config, f"on_epoch_histogram_{histo_type}_group")
                full_name = " / ".join([group, name])
                tf.summary.histogram(
                    full_name,
                    histo_values,
                    step=step,
                    buckets=self.config.testing_image_histogram_buckets,
                )

    def _make_epoch_histogram_images(self, step, histo_data):
        self.log.debug("-> Generating histogram images...")
        with self.histogram_writer.as_default():
            for img_name, img_data in self.joint_histograms.items():
                name = getattr(self.config, f"on_epoch_histogram_image_{img_name}_name")
                group = getattr(
                    self.config, f"on_epoch_histogram_image_{img_name}_group"
                )
                full_name = " / ".join([group, name])
                image = self._make_image_from_plot(
                    f"{name}, Epoch: {step}",
                    [histo_data[key] for key in img_data["histo_keys"]],
                    img_data["data_labels"],
                    img_data["data_colors"],
                    xlabel=getattr(
                        self.config, find_axis_label(img_data["histo_keys"][0]), ""
                    ),
                )
                tf.summary.image(full_name, image, step=step)

    def _make_epoch_energy_resolution(self, step, true_energy, pred_energy):
        self.log.debug("-> Generating energy resolution image...")
        with self.histogram_writer.as_default():
            name = self.config.on_epoch_image_energy_resolution_name
            group = self.config.on_epoch_image_energy_resolution_group
            full_name = " / ".join([group, name])
            image = self._make_image_from_plot(
                f"{name}, Epoch: {step}",
                [true_energy, pred_energy],
                xlabel="Energy [MeV]",
                ylabel=r"$\sigma$ / E [$\%$]",
                plot_type="energy_resolution",
            )
            tf.summary.image(full_name, image, step=step)

    def _make_epoch_histogram_events(self, step, tuples):
        self.log.debug("-> Generating gallery of events...")
        with self.histogram_writer.as_default():
            for evt, (hits, ys, preds) in enumerate(zip(*tuples)):
                name = f"Example {evt}"
                group = "Gallery"
                full_name = " / ".join([group, name])
                image = self._make_image_from_plot(
                    f"{name}, Epoch: {step}",
                    [hits, ys, preds],
                    xlabel=f"X {getattr(self.config, find_axis_label('x_pos'))}",
                    ylabel=f"Y {getattr(self.config, find_axis_label('y_pos'))}",
                    plot_type="event",
                )
                tf.summary.image(full_name, image, step=step)

    def _make_epoch_confusion_matrix(self, step, true_classes, pred_classes):
        self.log.debug("-> Generating confusion matrices...")
        with self.histogram_writer.as_default():
            name = "Confusion Matrix"
            group = "Classification"
            full_name = " / ".join([group, name])
            image = self._make_image_from_plot(
                f"{name}, Epoch: {step}",
                [true_classes, pred_classes],
                plot_type="confusion_matrix",
                title="",
                loc=0,
                fontsize=15,
            )
            tf.summary.image(full_name, image, step=step)

    def _make_epoch_performance_scalars(self, step, timings, data):
        self.log.debug("-> Generating performance & timing values/plots...")
        with self.histogram_writer.as_default():
            for timing_name, timing_value in timings.items():
                time_per_event = timing_value / len(data["true_position"]) * 1000.0
                tf.summary.scalar(
                    f"Timing/{timing_name}",
                    time_per_event,
                    step=step,
                )
                self.timings[timing_name].append(time_per_event)
            image = self._make_image_from_plot(
                f"Timing Summary, Epoch: {step}",
                [[range(len(values)), values] for values in self.timings.values()],
                self.timings.keys(),
                ["b", "g", "r", "c", "m"],
                xlabel="Epoch",
                ylabel="Time per event [ms]",
                plot_type="scatter",
            )
            tf.summary.image("Timing/Summary", image, step=step)

            data_types_names = {
                "true_x_pos": "Truth",
                "pred_x_pos": "Predicted",
                "matched_pred_x_pos": "Matched",
                "ghost_x_pos": "Ghost",
                "missed_x_pos": "Missed",
            }
            for data_type, data_name in data_types_names.items():
                clusters_no_tmp = len(ragged_to_normal(data[data_type].flatten()))
                self.log.debug(f"--> {data_name} Clusters No.: {clusters_no_tmp}")
                tf.summary.scalar(
                    f"Performance/{data_name}",
                    clusters_no_tmp,
                    step=step,
                )
                self.clusters_no[data_type].append(clusters_no_tmp)
            image = self._make_image_from_plot(
                f"Performance Summary, Epoch: {step}",
                [[range(len(values)), values] for values in self.clusters_no.values()],
                data_types_names.values(),
                ["b", "g", "r", "c", "m"],
                xlabel="Epoch",
                ylabel="Clusters No.",
                plot_type="scatter",
            )
            tf.summary.image("Performance/Summary", image, step=step)

    def on_epoch_end(self, epoch, logs=None):
        self.log.debug(f"Testing epoch {epoch} with {self.config.on_epoch_samples}.")
        times, tests = prepare_dataset_for_inference(
            self.config,
            self.image_transformation,
            self.model,
            self.dataset,
            self.config.on_epoch_samples,
        )

        self._make_epoch_performance_scalars(epoch, times, tests)

        gallery_events = [
            np.squeeze(tests["images"][:10]),
            tests["true_position"][:10, ..., :4],
            tests["pred_position"][:10, ..., :4],
        ]

        histo_types = {
            k: v for k, v in tests.items() if k not in self.non_automatic_histo_types
        }
        for histo_type, histo_values in histo_types.items():
            histo_types[histo_type] = ragged_to_normal(histo_values.flatten())

        self._make_epoch_confusion_matrix(
            epoch,
            ragged_to_normal(tests["matched_true_classes"].flatten()),
            ragged_to_normal(tests["matched_pred_classes"].flatten()),
        )
        self._make_epoch_histograms(epoch, histo_types)
        self._make_epoch_energy_resolution(
            epoch,
            ragged_to_normal(tests["matched_true_energy"].flatten()),
            ragged_to_normal(tests["matched_pred_energy"].flatten()),
        )
        self._make_epoch_histogram_images(epoch, histo_types)
        self._make_epoch_histogram_events(epoch, gallery_events)

        self.log.debug(f"Done.")
