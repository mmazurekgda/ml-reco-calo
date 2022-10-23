import numpy as np
import os
import json

class Config:

    TRAINING_OPTIONS = {
        'learning_rate': 1e-4,
        'epochs': 30,
        'batch_size': 1,
        # 'samples': 100,
        # 'validation_samples': 100,
        # 'validation_split': .2,
        'iou_ignore': .5,
        'dataset_cache': False,
        # 'shuffle_buffer_size': = 1000  # None = do not shuffle
        # early stopping
        'early_stopping': True,
        'early_stopping_patience': 6,
        'early_stopping_verbosity': 1,
        'early_stopping_restore': True,
        # reduce lr 
        'reduce_lr': True,
        'reduce_lr_verbosity': 1,
        'reduce_lr_patience': 5,
        'reduce_lr_cooldown': 5,
        # model checkpoint
        'model_checkpoint': True,
        'model_checkpoint_verbosity': 1,
        'model_checkpoint_out_weight_path': 'weights.tf',
        'model_checkpoint_save_weights_only': True,
        'model_checkpoint_save_best_only': True,
        # tensorboard
        # tensorboard_dir = '{}/logs'.format(conf_timestamp)
        'load_weight_path': None,

    }

    TFRECORDS_DATALOADER = {
        'tfrecords_files': '',
        'tfrecords_validation_files': '',
        'tfrecords_buffer_size': None,
        'tfrecords_num_parallel_reads': os.cpu_count(),
        'tfrecords_compression_type': 'ZLIB',
    }

    INFERENCE_OPTIONS = {
        'max_boxes': 1500,
        'classes': ['one'],
        'iou_threshold': .5,
        'score_threshold': .5,
        'soft_nms_sigma': .5,
    }

    DATA_OPTIONS = {
        'img_width': 384,
        'img_height': 312,
        'target_img_width': 384,
        'target_img_height': 312,
        'channels': 1,
        'energy_cols_no': 1,
    }

    CNN_OPTIONS = {
        'anchors_not_parsed': np.array([
            (6, 6),
            (9, 9),
            (18, 18),
        ], np.float32),
        'anchor_masks': np.array([
            (6, 6),
            (9, 9),
            (18, 18),
        ], np.float32),
        'granularities': [
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
        'anchors': np.ndarray([]),
        'classes_no': 0,
        'features_no': 0,
    }

    OPTIONS = {
        **CONFIGURABLE_OPTIONS,
        **NON_CONFIGURABLE_OPTIONS,
    }

    _frozen = False

    def __setattr__(self, key, value):
        if self._frozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def __init__(self, load_config_file=None):
        if load_config_file:
            with open(load_config_file) as json_dump:
                data = json.load(json_dump)
                for prop, value in json.loads(data).items():
                    setattr(self, prop, self._safe_object(prop, value))
        else:
            for prop, value in self.OPTIONS.items():
                setattr(self, prop, value)
        self._frozen = True
        if not load_config_file:
            self.anchors = self.anchors_not_parsed / [self.img_width, self.img_height]
            self.classes_no = len(self.classes) if isinstance(self.classes, list) else 1
            self.features_no = 4 + self.energy_cols_no + self.classes_no

    def _safe_JSON(self, value):
        if type(value) is np.ndarray:
            return value.tolist()
        return value

    def _safe_object(self, key, value):
        if key in ['anchors_not_parsed', 'anchors_masks', 'anchors']:
            return np.array(value)
        return value

    def to_JSON(self):
        options_copy = { key: self._safe_JSON(getattr(self, key)) for key in self.OPTIONS }
        return json.dumps(options_copy, sort_keys=True, indent=4)

    def dump_to_file(self, config_file):
        with open(config_file, 'w') as json_dump:
            json.dump(self.to_JSON(), json_dump)

    # FIXME: workaround for methods defined in core
    def refine_boxes(self, pred, anchors):
        raise NotImplementedError

    # FIXME: workaround for methods defined in core
    def nms(self, outputs):
        raise NotImplementedError
