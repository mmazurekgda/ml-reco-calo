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

    data_types_names = {
        "true_x_pos": "Truth",
        "pred_x_pos": "Predicted",
        "matched_pred_x_pos": "Matched",
        "ghost_x_pos": "Ghost",
        "missed_x_pos": "Missed",
    }

    joint_histograms = {
        "comparison_energy": {
            "histo_keys": ["true_energy", "pred_energy"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "comparison_cluster_width": {
            "histo_keys": ["true_width", "pred_width"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "comparison_cluster_height": {
            "histo_keys": ["true_height", "pred_height"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "comparison_cluster_x_pos": {
            "histo_keys": ["true_x_pos", "pred_x_pos"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "comparison_cluster_x_pos_extended": {
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
        "comparison_cluster_y_pos": {
            "histo_keys": ["true_y_pos", "pred_y_pos"],
            "data_labels": ["Truth", "Predicted"],
            "data_colors": ["blue", "red"],
        },
        "comparison_cluster_y_pos_extended": {
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
            if self.config.testing_image_label_rlabel_infrastructure:
                rlabel = " | ".join(
                    [
                        rlabel,
                        self.config.testing_image_label_rlabel_infrastructure,
                    ]
                )
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

    def _make_histograms(self, step, histo_data):
        self.log.debug("-> Generating histograms...")
        with self.histogram_writer.as_default():
            for histo_type, histo_values in histo_data.items():
                name = getattr(self.config, f"on_callback_histogram_{histo_type}_name")
                group = getattr(
                    self.config, f"on_callback_histogram_{histo_type}_group"
                )
                full_name = " / ".join([f"{self.prefix} {group}", name])
                tf.summary.histogram(
                    full_name,
                    histo_values,
                    step=step,
                    buckets=self.config.testing_image_histogram_buckets,
                )

    def _make_histogram_images(self, step, step_name, histo_data):
        self.log.debug("-> Generating histogram images...")
        with self.histogram_writer.as_default():
            for img_name, img_data in self.joint_histograms.items():
                name = getattr(
                    self.config, f"on_callback_histogram_image_{img_name}_name"
                )
                group = getattr(
                    self.config, f"on_callback_histogram_image_{img_name}_group"
                )
                full_name = " / ".join([f"{self.prefix} {group}", name])
                image = self._make_image_from_plot(
                    f"{name}, {step_name}",
                    [histo_data[key] for key in img_data["histo_keys"]],
                    img_data["data_labels"],
                    img_data["data_colors"],
                    xlabel=getattr(
                        self.config, find_axis_label(img_data["histo_keys"][0]), ""
                    ),
                )
                tf.summary.image(full_name, image, step=step)

    def _make_energy_resolution(self, step, step_name, true_energy, pred_energy):
        self.log.debug("-> Generating energy resolution image...")
        with self.histogram_writer.as_default():
            name = self.config.on_callback_image_energy_resolution_name
            group = self.config.on_callback_image_energy_resolution_group
            full_name = " / ".join([f"{self.prefix} {group}", name])
            image = self._make_image_from_plot(
                f"{name}, {step_name}",
                [true_energy, pred_energy],
                xlabel="Energy [MeV]",
                ylabel=r"$\sigma$ / E [$\%$]",
                plot_type="energy_resolution",
            )
            tf.summary.image(full_name, image, step=step)

    def _make_histogram_events(self, step, step_name, tuples):
        self.log.debug("-> Generating gallery of events...")
        with self.histogram_writer.as_default():
            for evt, (hits, ys, preds) in enumerate(zip(*tuples)):
                name = f"Example {evt}"
                group = "Gallery"
                full_name = " / ".join([f"{self.prefix} {group}", name])
                image = self._make_image_from_plot(
                    f"{name}, {step_name}",
                    [hits, ys, preds],
                    xlabel=f"X {getattr(self.config, find_axis_label('x_pos'))}",
                    ylabel=f"Y {getattr(self.config, find_axis_label('y_pos'))}",
                    plot_type="event",
                )
                tf.summary.image(full_name, image, step=step)

    def _make_confusion_matrix(self, step, step_name, true_classes, pred_classes):
        self.log.debug("-> Generating confusion matrices...")
        with self.histogram_writer.as_default():
            name = "Confusion Matrix"
            group = "Classification"
            full_name = " / ".join([f"{self.prefix} {group}", name])
            image = self._make_image_from_plot(
                f"{name}, {step_name}",
                [true_classes, pred_classes],
                plot_type="confusion_matrix",
                title="",
                loc=0,
                fontsize=15,
            )
            tf.summary.image(full_name, image, step=step)

    def _make_performance_scalars(self, step, step_name):
        self.log.debug("-> Generating performance & timing values/plots...")
        with self.histogram_writer.as_default():
            for timing_name, time_per_event in self.timings.items():
                tf.summary.scalar(
                    f"{self.prefix} Timing/{timing_name}",
                    time_per_event[-1],
                    step=step,
                )
            image = self._make_image_from_plot(
                f"Timing Summary, {step_name}",
                [[range(len(values)), values] for values in self.timings.values()],
                self.timings.keys(),
                ["b", "g", "r", "c", "m"],
                xlabel="Epoch",
                ylabel="Time per event [ms]",
                plot_type="scatter",
            )
            tf.summary.image(f"{self.prefix} Timing/Summary", image, step=step)

            for data_name, clusters_no in self.clusters_no.items():
                tf.summary.scalar(
                    f"{self.prefix} Performance/{data_name}",
                    clusters_no[-1],
                    step=step,
                )
            image = self._make_image_from_plot(
                f"Performance Summary, {step_name}",
                [[range(len(values)), values] for values in self.clusters_no.values()],
                self.clusters_no.keys(),
                ["b", "g", "r", "c", "m"],
                xlabel="Epoch",
                ylabel="Clusters No.",
                plot_type="scatter",
            )
            tf.summary.image(f"{self.prefix} Performance/Summary", image, step=step)

    def _plot(self, step, step_name):
        self._make_performance_scalars(step, step_name)

        gallery_events = [
            np.squeeze(self.tests["images"][:10]),
            self.tests["true_position"][:10, ..., :4],
            self.tests["pred_position"][:10, ..., :4],
        ]

        histo_types = {
            k: v
            for k, v in self.tests.items()
            if k not in self.non_automatic_histo_types
        }
        for histo_type, histo_values in histo_types.items():
            histo_types[histo_type] = ragged_to_normal(histo_values.flatten())

        self._make_confusion_matrix(
            step,
            step_name,
            ragged_to_normal(self.tests["matched_true_classes"].flatten()),
            ragged_to_normal(self.tests["matched_pred_classes"].flatten()),
        )
        self._make_histograms(step, histo_types)
        self._make_histogram_images(step, step_name, histo_types)
        self._make_energy_resolution(
            step,
            step_name,
            ragged_to_normal(self.tests["matched_true_energy"].flatten()),
            ragged_to_normal(self.tests["matched_pred_energy"].flatten()),
        )
        self._make_histogram_events(step, step_name, gallery_events)

        self.log.debug(f"Done.")

    def _get_setup_performance(self, **kwargs):
        self.log.info(f"-> Checking for \n'{kwargs}'")
        self.config._unfreeze()
        # -> set the options here
        for option, value in kwargs.items():
            setattr(self.config, option, value)
        self.config._freeze()
        self.times, self.tests = prepare_dataset_for_inference(
            self.config,
            self.image_transformation,
            self.model,
            self.dataset,
            self.config.on_train_end_samples,
        )
        self._check_performance()

    def _check_performance(self):
        self.log.debug("--> Checking performance...")
        longest_txt = max([len(key) for key in self.times.keys()])
        for timing_name, timing_value in self.times.items():
            time_per_event = timing_value / len(self.tests["true_position"]) * 1000.0
            self.timings[timing_name].append(time_per_event)
            msg = f"---> {timing_name} time:{' ' * (longest_txt - len(timing_name))} {round(timing_value, 3)} s,"
            msg += f" {round(time_per_event, 3)} ms/event"
            msg += f" {round(timing_value / self.times['Total'] * 100, 3)} % total time"
            self.log.debug(msg)

        for data_type, data_name in self.data_types_names.items():
            clusters_no_tmp = len(ragged_to_normal(self.tests[data_type].flatten()))
            self.log.debug(f"---> {data_name} Clusters No.: {clusters_no_tmp}")
            self.clusters_no[data_name].append(clusters_no_tmp)

    def _calibrate(self):
        self.log.info("Calibration started...")
        self.timings.clear()
        self.clusters_no.clear()
        tested_opts = []
        for iou_th in self.config.calibrate_iou_threshold_values:
            for score_th in self.config.calibrate_score_threshold_values:
                for soft_nms_th in self.config.calibrate_soft_nms_sigma_values:
                    opts = dict(
                        iou_threshold=iou_th,
                        score_threshold=score_th,
                        soft_nms_sigma=soft_nms_th,
                    )
                    tested_opts.append(opts)
                    self._get_setup_performance(**opts)
        self.log.info("Maximizing over: '{self.config.calibrate_measure}'")
        measure_values = {}
        if self.config.calibrate_measure == "f-score":
            measure_values = [
                (
                    opts,
                    2
                    * self.clusters_no["Matched"][i]
                    / (
                        2 * self.clusters_no["Matched"][i]
                        + self.clusters_no["Ghost"][i]
                        + self.clusters_no["Missed"][i]
                    ),
                )
                for i, opts in enumerate(tested_opts)
            ]
            measure_values = sorted(measure_values, key=lambda x: x[1])
            self.log.debug(f"-> F1 scores:")
            for measure_value in measure_values:
                self.log.debug(f"--> {measure_value[0]}: {measure_value[1]}")
        else:
            raise NotImplementedError()
        best_options = measure_values[-1]
        self.clusters_no.clear()
        self.timings.clear()
        self.log.info(f"-> Best option is: \n{best_options[0]}")
        self.log.debug(f"-> Setting the best option for further inference.")
        self.config._unfreeze()
        for option, value in best_options[0].items():
            setattr(self.config, option, value)
        self.config._freeze()


class CNNTestingAtTrainingCallback(CNNTestingCallback):
    def on_epoch_end(self, epoch, logs=None):
        self.prefix = "Monitoring"
        samples = self.config.on_epoch_end_samples
        samples_msg = f"{samples} samples"
        if samples == self.config.test_samples:
            samples_msg = " (the whole testing dataset)"
        self.log.info(f"Monitoring of training for epoch {epoch} with {samples_msg}...")
        self.times, self.tests = prepare_dataset_for_inference(
            self.config,
            self.image_transformation,
            self.model,
            self.dataset,
            samples,
        )
        self._check_performance()
        self._plot(epoch, f"Epoch: {epoch}")

    def on_train_end(self, logs=None):
        self.prefix = "Inference"
        step_name = "Final Not Calibrated"
        samples = self.config.on_train_end_samples
        samples_msg = f"{samples} samples"
        if samples == self.config.test_samples:
            samples_msg += " (the whole testing dataset)"
        self.log.info(f"Final inference with {samples_msg}...")
        if self.config.calibrate:
            step_name = "Final Calibrated"
            self._calibrate()
        self.times, self.tests = prepare_dataset_for_inference(
            self.config,
            self.image_transformation,
            self.model,
            self.dataset,
            samples,
        )
        self._check_performance()
        self._plot(0, step_name)
