import numpy as np
import os
import json
import logging
import git
import copy


class Config:

    TRAINING_OPTIONS = {
        "learning_rate": 1e-4,
        "epochs": 30,
        "batch_size": 1,
        "samples": -1,
        "validation_samples": -1,
        "test_samples": -1,
        # 'validation_split': .2,
        "iou_ignore": 0.5,
        "dataset_cache": True,
        # "dataset_repeat": True,
        # 'shuffle_buffer_size': = 1000  # None = do not shuffle
        # early stopping
        "early_stopping": True,
        "early_stopping_patience": 6,
        "early_stopping_verbosity": 1,
        "early_stopping_restore": True,
        # reduce lr
        "reduce_lr": True,
        "reduce_lr_verbosity": 1,
        "reduce_lr_patience": 5,
        "reduce_lr_cooldown": 5,
        # model checkpoint
        "model_checkpoint": True,
        "model_checkpoint_verbosity": 1,
        "model_checkpoint_out_weight_file": "weights.tf",
        "model_checkpoint_save_weights_only": True,
        "model_checkpoint_save_best_only": True,
        # tensorboard
        "tensorboard": False,
        "tensorboard_log_dir": "tensorboard",
        "tensorboard_histogram_freq": 1,
        "tensorboard_write_graph": True,
        "tensorboard_write_images": True,
        # "tensorboard_write_steps_per_second": False,
        # -> not available in TF 2.0
        "tensorboard_update_freq": "epoch",
        "tensorboard_profile_batch": 0,
        "tensorboard_embeddings_freq": 1,
        "tensorboard_embeddings_metadata": None,
        # other
        "load_weight_path": None,
    }

    TFRECORDS_DATALOADER = {
        "tfrecords_files": "",
        "tfrecords_validation_files": "",
        "tfrecords_test_files": "",
        "tfrecords_buffer_size": None,
        # "tfrecords_num_parallel_reads": os.cpu_count(),
        "tfrecords_compression_type": "ZLIB",
    }

    INFERENCE_OPTIONS = {
        "max_boxes": 1500,
        "classes": ["one"],
        "iou_threshold": 0.5,
        # the two below should investigated during validation
        "score_threshold": float("-inf"),
        # the two below should investigated during validation
        "soft_nms_sigma": 0.0,
        "calibrate": True,
        "calibrate_measure": "f-score",
        "calibrate_iou_threshold_values": [0.0, 0.25, 0.5, 0.75, 0.9],
        "calibrate_score_threshold_values": [0.0, 0.25, 0.5, 0.75, 0.9],
        "calibrate_soft_nms_sigma_values": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
    }

    TEST_OPTIONS = {
        "testing": True,
        # general labels
        "testing_image_label_exp": "Gaussino",
        "testing_image_label_llabel": "Simulation Preliminary",
        "testing_image_label_rlabel": "",
        "testing_image_label_rlabel_infrastructure": "",
        "testing_image_figure_x_size": 15,
        "testing_image_figure_y_size": 10,
        "testing_image_histogram_buckets": 100,
        # histograms
        "on_callback_histogram_writer_name": "clusterization",
        "on_callback_histogram_true_energy_name": "True Energy",
        "on_callback_histogram_true_energy_group": "Energy",
        "on_callback_histogram_pred_energy_name": "Predicted Energy",
        "on_callback_histogram_pred_energy_group": "Energy",
        "on_callback_histogram_true_width_name": "True Cluster Width",
        "on_callback_histogram_true_width_group": "Cluster Width",
        "on_callback_histogram_pred_width_name": "Predicted Cluster Width",
        "on_callback_histogram_pred_width_group": "Cluster Width",
        "on_callback_histogram_true_height_name": "True Cluster Height",
        "on_callback_histogram_true_height_group": "Cluster Height",
        "on_callback_histogram_pred_height_name": "Predicted Cluster Height",
        "on_callback_histogram_pred_height_group": "Cluster Height",
        "on_callback_histogram_true_x_pos_name": "True Cluster X Position",
        "on_callback_histogram_true_x_pos_group": "Cluster X Position",
        "on_callback_histogram_matched_true_x_pos_name": "Matched True Cluster X Position",
        "on_callback_histogram_matched_true_x_pos_group": "Cluster X Position",
        "on_callback_histogram_pred_x_pos_name": "Predicted Cluster X Position",
        "on_callback_histogram_pred_x_pos_group": "Cluster X Position",
        "on_callback_histogram_matched_pred_x_pos_name": "Matched Predicted Cluster X Position",
        "on_callback_histogram_matched_pred_x_pos_group": "Cluster X Position",
        "on_callback_histogram_true_y_pos_name": "True Cluster Y Position",
        "on_callback_histogram_true_y_pos_group": "Cluster Y Position",
        "on_callback_histogram_matched_true_y_pos_name": "Matched True Cluster Y Position",
        "on_callback_histogram_matched_true_y_pos_group": "Cluster Y Position",
        "on_callback_histogram_pred_y_pos_name": "Predicted Cluster Y Position",
        "on_callback_histogram_pred_y_pos_group": "Cluster Y Position",
        "on_callback_histogram_matched_pred_y_pos_name": "Matched Predicted Cluster Y Position",
        "on_callback_histogram_matched_pred_y_pos_group": "Cluster Y Position",
        "on_callback_histogram_score_name": "Score",
        "on_callback_histogram_score_group": "Score",
        # labels
        "on_callback_histogram_energy_label": "Energy [MeV]",
        "on_callback_histogram_pos_label": "Position [mm]",
        "on_callback_histogram_length_label": "Length [mm]",
        # images
        "on_callback_histogram_image_comparison_energy_name": "Particle Energy",
        "on_callback_histogram_image_comparison_energy_group": "Energy",
        "on_callback_histogram_image_comparison_cluster_width_name": "Cluster Width",
        "on_callback_histogram_image_comparison_cluster_width_group": "Cluster Width",
        "on_callback_histogram_image_comparison_cluster_height_name": "Cluster Height",
        "on_callback_histogram_image_comparison_cluster_height_group": "Cluster Height",
        "on_callback_histogram_image_comparison_cluster_x_pos_name": "Cluster X Position",
        "on_callback_histogram_image_comparison_cluster_x_pos_group": "Cluster X Position",
        "on_callback_histogram_image_comparison_cluster_x_pos_extended_name": "Cluster X Position (Extended)",
        "on_callback_histogram_image_comparison_cluster_x_pos_extended_group": "Cluster X Position",
        "on_callback_histogram_image_comparison_cluster_y_pos_name": "Cluster Y Position",
        "on_callback_histogram_image_comparison_cluster_y_pos_group": "Cluster Y Position",
        "on_callback_histogram_image_comparison_cluster_y_pos_extended_name": "Cluster Y Position (Extended)",
        "on_callback_histogram_image_comparison_cluster_y_pos_extended_group": "Cluster Y Position",
        "on_callback_image_energy_resolution_name": "Energy Resolution",
        "on_callback_image_energy_resolution_group": "Energy",
        # needed for the position converter,
        "img_x_max": -1,
        "img_x_min": -1,
        "img_y_max": -1,
        "img_y_min": -1,
        # needed for the hit converter
        "max_hit_energy": -1,
        "min_hit_energy": -1,
        # needed for the energy converter,
        "convert_energy": "normalize",  # or standardize
        "std_particle_energy": -1,
        "mean_particle_energy": -1,
        "max_particle_energy": -1,
        "min_particle_energy": -1,
    }

    DATA_OPTIONS = {
        "img_width": 384,
        "img_height": 312,
        "target_img_width": 384,
        "target_img_height": 312,
        "channels": 1,
        "energy_cols_no": 1,
    }

    CNN_OPTIONS = {
        "anchors_not_parsed": np.array(
            [
                (6, 6),
                (9, 9),
                (18, 18),
            ],
            np.float32,
        ),
        "anchor_masks": np.array(
            [
                (6, 6),
                (9, 9),
                (18, 18),
            ],
            np.float32,
        ),
        "granularities": [
            (8, 8),
            (4, 4),
            (2, 2),
        ],
    }

    CONFIGURABLE_OPTIONS = {
        **TRAINING_OPTIONS,
        **TFRECORDS_DATALOADER,
        **INFERENCE_OPTIONS,
        **TEST_OPTIONS,
        **DATA_OPTIONS,
        **CNN_OPTIONS,
    }

    NON_CONFIGURABLE_OPTIONS = {
        "anchors": np.ndarray([]),
        "classes_no": 0,
        "input_features_no": 0,
    }

    OPTIONS = {
        **CONFIGURABLE_OPTIONS,
        **NON_CONFIGURABLE_OPTIONS,
    }

    _frozen = False
    _rigid = False

    _options_with_dirs = [
        "tfrecords_files",
        "tfrecords_validation_files",
        "tfrecords_test_files",
        "load_weight_path",
    ]

    _output_area_dirs = [
        "model_checkpoint_out_weight_file",
        "tensorboard_log_dir",
    ]

    def __setattr__(self, key, value):
        new_value = copy.copy(value)  # FIXME: needed? or too paranoid
        if self._frozen and key != "_frozen":
            msg = f"Config became a frozen class. No futher changes possible. Tried setting '{key}': '{value}'"
            if hasattr(self, "log"):
                self.log.error(msg)
            raise TypeError(msg)
        if self._rigid and not hasattr(self, key):
            msg = f"Config became a rigid class. No additional options possible. Tried setting '{key}': '{value}'"
            if hasattr(self, "log"):
                self.log.error(msg)
            raise TypeError(msg)
        if key in self.OPTIONS:
            if key in self._output_area_dirs and new_value:
                new_value = "/".join([self.output_area, new_value])
            if key in self._options_with_dirs and new_value:
                new_value = self.change_to_local_paths(key, value)
            override_text = "DEFAULT"
            if self._rigid:
                override_text = "NEW VALUE"
                if not self._safe_JSON(new_value):
                    self.log.warning(f"--> Setting an undefined property for {key}")
            self.log.debug(f"--> ({override_text}) '{key}': {new_value}")
        object.__setattr__(self, key, new_value)
        if self._rigid:
            self._check_for_non_configurables(key=key)

    def _check_for_non_configurables(self, key="", all=False):
        if all or key in ["anchors_not_parsed", "img_width", "img_height"]:
            self.anchors = self.anchors_not_parsed / [self.img_width, self.img_height]
        if all or key in ["classes"]:
            self.classes_no = len(self.classes) if isinstance(self.classes, list) else 1
        if all or key in ["energy_classes_no"]:
            self.input_features_no = 4 + self.energy_cols_no + 1

    def __init__(self, output_area="./", load_config_file=None, freeze=False):
        self.log = logging.getLogger("MCRecoCalo")
        self.log.info(f"Initialized a new config.")
        self._set_working_area()
        self._set_output_area(output_area)
        self.log.debug(f"-> Loading options from file: {load_config_file}.")
        for prop, value in self.OPTIONS.items():
            setattr(self, prop, value)
        # disable adding new members in case the config file is not compatible
        self._rigidify()
        if load_config_file:
            self.log.debug(f"-> Loading options from file: {load_config_file}.")
            with open(load_config_file) as json_dump:
                data = json.load(json_dump)
                for prop, value in json.loads(data).items():
                    if prop not in self._output_area_dirs:
                        parsed_value = self._safe_object(prop, value)
                        setattr(self, prop, parsed_value)
        else:
            self._check_for_non_configurables(all=True)
            self.log.warning("-> No config file given. Will use the default values.")
        if freeze:
            self._freeze()

    def _set_working_area(self):
        repo = git.Repo(".", search_parent_directories=True)
        if (
            repo.remotes.origin.url.split(".git")[0].split("/")[-1]
            == "ml-reco-calo-datasets"
        ):
            repo = git.Repo(
                repo.working_tree_dir + "/..", search_parent_directories=True
            )
        if repo.remotes.origin.url.split(".git")[0].split("/")[-1] != "ml-reco-calo":
            msg = "Invalid working area. Must be inside MLRecoCalo repository."
            log.error(msg)
            raise ValueError(msg)
        self.working_area = repo.working_tree_dir
        self.log.debug(f"-> Working area is: {self.working_area}")

    def _set_output_area(self, directory):
        if not directory:
            self.output_area = "./"
        elif not os.path.exists(directory):
            os.makedirs(directory)
        self.output_area = directory
        self.log.debug(f"-> Output area is: {self.output_area}")

    def change_to_local_paths(self, option, paths):
        if type(paths) is str:
            return self.ensure_local_path(option, paths)
        elif hasattr(paths, "__iter__"):
            new_paths = []
            for path in paths:
                new_paths.append(self.ensure_local_path(option, path))
            return new_paths
        else:
            msg = f"-> Invalid options '{option}': '{paths}'. Must be string or an iterable."
            self.log.error(msg)
            raise TypeError(msg)

    def ensure_local_path(self, option, path):
        local_path = ""
        if os.path.isabs(path):
            if not os.path.exists(path):
                msg = f"-> The '{option}' has an invalid path. The file '{path}' does not exist."
                self.log.error(msg)
                raise FileNotFoundError(msg)
            if self.working_area not in path:
                msg = f"-> The '{option}' has an invalid path. Must contain the working area: '{self.working_area}'"
                self.log.error(msg)
                raise FileNotFoundError(msg)
            return path.replace(self.working_area + "/", "")
        else:
            if not os.path.exists("/".join([self.working_area, path])):
                msg = f"-> The '{option}' has an invalid path. Must be with respect to the working area: '{self.working_area}'"
                self.log.error(msg)
                raise FileNotFoundError(msg)
            return path

    def paths_to_global(self, paths):
        directory = self.working_area
        if type(paths) is str:
            return "/".join([directory, paths])
        if hasattr(paths, "__iter__"):
            new_paths = []
            for path in paths:
                new_paths.append("/".join([directory, path]))
            return new_paths

    def _rigidify(self):
        self._rigid = True
        self.log.debug("-> Making the options rigid. No additional members possible.")

    def _unrigidify(self):
        self._rigid = False
        self.log.debug("-> Making the options flexible. Additional members possible.")

    def _freeze(self):
        self._frozen = True
        self.log.debug("-> Freezing options. No additional changes possible.")

    def _unfreeze(self):
        self._frozen = False
        self.log.debug("-> Unfreezing options. Additional changes possible.")

    def _safe_JSON(self, value):
        if type(value) is np.ndarray:
            return value.tolist()
        return value

    def _safe_object(self, key, value):
        if key in ["anchors_not_parsed", "anchors_masks", "anchors"]:
            return np.array(value)
        return value

    def to_JSON(self):
        options_copy = {
            key: self._safe_JSON(getattr(self, key)) for key in self.OPTIONS
        }
        return json.dumps(options_copy, sort_keys=True, indent=4)

    def dump_to_file(self, config_file="config.json"):
        config_path = "/".join([self.output_area, config_file])
        with open(config_path, "w") as json_dump:
            json.dump(self.to_JSON(), json_dump)
        self.log.debug(f"-> Config dumped to '{config_path}'")

    def set_options(self, options: dict = {}, permit_when_frozen=False):
        was_frozen = bool(self._frozen)
        if was_frozen and permit_when_frozen:
            self._unfreeze()
        for option, value in options.items():
            self.log.debug(f"-> Setting option {option}: {value}.")
            setattr(self, option, value)
        if was_frozen:
            self._freeze()

    def check_compatibility(self):
        msg = ""
        if not self.convert_energy in ["normalize", "standardize"]:
            msg = "You can either normalize or standardize energies"
        if msg:
            self.log.error(msg)
            raise ValueError(msg)

    def convert_to_energy(self, energy):
        if self.convert_energy == "standardize":
            return energy * self.std_particle_energy + self.mean_particle_energy
        if self.convert_energy == "normalize":
            return (
                energy * (self.max_particle_energy - self.min_particle_energy)
                + self.min_particle_energy
            )
        raise NotImplementedError()

    def convert_to_hit_energy(self, energy):
        return (
            energy * (self.max_hit_energy - self.min_hit_energy) + self.min_hit_energy
        )

    def convert_to_position(self, position, dim="x"):
        if dim == "x":
            return position * (self.img_x_max - self.img_x_min) + self.img_x_min
        if dim == "y":
            return position * (self.img_y_max - self.img_y_min) + self.img_y_min
        raise NotImplementedError()

    # FIXME: workaround for methods defined in core
    def refine_boxes(self, pred, anchors):
        raise NotImplementedError

    # FIXME: workaround for methods defined in core
    def nms(self, outputs):
        raise NotImplementedError
