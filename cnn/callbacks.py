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
    find_axis_label,
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

    def _make_image_from_plot(
        self,
        name,
        data_tuple,
        data_labels,
        data_colors,
        xlabel="",
        ylabel="",
        plot_type="hist",
    ):
        plt.style.use(hep.style.LHCb2)
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(
                self.config.on_epoch_histogram_image_figure_x_size,
                self.config.on_epoch_histogram_image_figure_y_size,
            ),
        )
        if plot_type == "hist":
            plot_histograms(
                ax,
                data_tuple,
                data_labels,
                data_colors,
                bins=self.config.on_epoch_histogram_buckets,
            )
        elif plot_type == "scatter":
            plot_scatter_plots(
                ax,
                data_tuple,
                data_labels,
                data_colors,
            )
        else:
            raise NotImplementedError()

        plt.title(name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def _make_epoch_histograms(self, step, histo_data):
        with self.histogram_writer.as_default():
            for histo_type, histo_values in histo_data.items():
                name = getattr(self.config, f"on_epoch_histogram_{histo_type}_name")
                group = getattr(self.config, f"on_epoch_histogram_{histo_type}_group")
                full_name = " / ".join([group, name])
                tf.summary.histogram(
                    full_name,
                    histo_values,
                    step=step,
                    buckets=self.config.on_epoch_histogram_buckets,
                    description=getattr(
                        self.config, f"on_epoch_histogram_{histo_type}_description"
                    ),
                )

    def _make_epoch_histogram_images(self, step, histo_data, img_keys):
        with self.histogram_writer.as_default():
            for img_name, img_data in img_keys.items():
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

    def _make_epoch_performance_scalars(self, step, timings, data):
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

    def on_epoch_end(self, epoch, logs=None):
        self.log.debug(f"Testing epoch {epoch} with {self.config.on_epoch_samples}.")
        times, tests = prepare_dataset_for_inference(
            self.log,
            self.image_transformation,
            self.model,
            self.dataset,
            self.config.on_epoch_samples,
        )

        convert_data(self.config, tests)

        self._make_epoch_performance_scalars(epoch, times, tests)

        histo_data = {
            "true_energy": tests["true_energy"],
            "pred_energy": tests["pred_energy"],
            "true_width": tests["true_position"][..., 2]
            - tests["true_position"][..., 0],
            "pred_width": tests["pred_position"][..., 2]
            - tests["pred_position"][..., 0],
            "true_height": tests["true_position"][..., 3]
            - tests["true_position"][..., 1],
            "pred_height": tests["pred_position"][..., 3]
            - tests["pred_position"][..., 1],
            "true_x_pos": (
                tests["true_position"][..., 2] + tests["true_position"][..., 0]
            )
            / 2.0,
            "pred_x_pos": (
                tests["pred_position"][..., 2] + tests["pred_position"][..., 0]
            )
            / 2.0,
            "true_y_pos": (
                tests["true_position"][..., 3] + tests["true_position"][..., 1]
            )
            / 2.0,
            "pred_y_pos": (
                tests["pred_position"][..., 3] + tests["pred_position"][..., 1]
            )
            / 2.0,
        }

        img_keys = {
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
            "vs_cluster_y_pos": {
                "histo_keys": ["true_y_pos", "pred_y_pos"],
                "data_labels": ["Truth", "Predicted"],
                "data_colors": ["blue", "red"],
            },
        }

        for histo_type, histo_values in histo_data.items():
            histo_data[histo_type] = ragged_to_normal(histo_values.flatten())

        self._make_epoch_histograms(epoch, histo_data)
        self._make_epoch_histogram_images(epoch, histo_data, img_keys)
        self.log.debug(f"Done.")
