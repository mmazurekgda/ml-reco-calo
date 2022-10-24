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
        # 'samples': 100,
        # 'validation_samples': 100,
        # 'validation_split': .2,
        "iou_ignore": 0.5,
        "dataset_cache": True,
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
        # tensorboard_dir = '{}/logs'.format(conf_timestamp)
        "load_weight_path": None,
    }

    TFRECORDS_DATALOADER = {
        "tfrecords_files": "",
        "tfrecords_validation_files": "",
        "tfrecords_buffer_size": None,
        "tfrecords_num_parallel_reads": os.cpu_count(),
        "tfrecords_compression_type": "ZLIB",
    }

    INFERENCE_OPTIONS = {
        "max_boxes": 1500,
        "classes": ["one"],
        "iou_threshold": 0.5,
        "score_threshold": 0.5,
        "soft_nms_sigma": 0.5,
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
        **DATA_OPTIONS,
        **CNN_OPTIONS,
    }

    NON_CONFIGURABLE_OPTIONS = {
        "anchors": np.ndarray([]),
        "classes_no": 0,
        "features_no": 0,
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
        "load_weight_path",
    ]

    _output_area_dirs = [
        "model_checkpoint_out_weight_file",
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
            self.log.warning("-> No config file given. Will use the default values.")
        if not load_config_file:
            self.anchors = self.anchors_not_parsed / [self.img_width, self.img_height]
            self.classes_no = len(self.classes) if isinstance(self.classes, list) else 1
            self.features_no = 4 + self.energy_cols_no + self.classes_no
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
            if not os.path.isfile(path):
                msg = f"-> The '{option}' has an invalid path. The file '{path}' does not exist."
                self.log.error(msg)
                raise FileNotFoundError(msg)
            if self.working_area not in path:
                msg = f"-> The '{option}' has an invalid path. Must contain the working area: '{self.working_area}'"
                self.log.error(msg)
                raise FileNotFoundError(msg)
            return path.replace(self.working_area + "/", "")
        else:
            if not os.path.isfile("/".join([self.working_area, path])):
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
                new_path.append("/".join([directory, path]))
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

    # FIXME: workaround for methods defined in core
    def refine_boxes(self, pred, anchors):
        raise NotImplementedError

    # FIXME: workaround for methods defined in core
    def nms(self, outputs):
        raise NotImplementedError
