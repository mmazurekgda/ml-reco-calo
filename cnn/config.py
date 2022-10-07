import numpy as np

class Config:

    TRAINING_OPTIONS = {
        'learning_rate': 1e-4,
        'epochs': 30,
        'batch_size': 1,
        'samples': 100,
        'validation_samples': 100,
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
        'model_checkpoint_out_weight_path': 'weights.tf',
        'model_checkpoint_save_weights_only': True,
        'model_checkpoint_save_best_only': True,
        # tensorboard
        # tensorboard_dir = '{}/logs'.format(conf_timestamp)
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

    OPTIONS = {
        **TRAINING_OPTIONS,
        **INFERENCE_OPTIONS,
        **DATA_OPTIONS,
        **CNN_OPTIONS,
    }

    def __init__(self):
        for prop, value in self.OPTIONS.items():
            setattr(self, prop, value)
        self.anchors = self.anchors_not_parsed / [self.img_width, self.img_height]
        self.classes_no = len(self.classes) if isinstance(self.classes, list) else 1

    '''
    def normalize_image(x):
        return (x - Config.digits_mean) / Config.digits_std
        #return tf.where(tf.less(x - Config.min_digit, 1.0), 0.0, ((x - Config.min_digit) / math.log(10.0)) / (Config.max_digit - Config.min_digit))
        #return tf.where(tf.less(x - Config.min_digit, 0.0), 0.0, ((x - Config.min_digit) / (Config.max_digit - Config.min_digit)))
        #return tf.where(tf.less(x, Config.min_digit), 1.0,  tf.math.log(x - Config.min_digit + 1.) / math.log(10.) / math.log10(Config.max_digit - Config.min_digit + 1.))
        #return tf.where(tf.less(x, Config.min_digit), 0.0,  tf.math.log(x - Config.min_digit + 1.) / math.log(10.) / math.log10(Config.max_digit - Config.min_digit + 1.))

    def transform_energy(energy):
        return (tf.convert_to_tensor(energy) - Config.particles_mean) / Config.particles_std
        #return (tf.convert_to_tensor(energy) - Config.min_energy) / (Config.max_energy - Config.min_energy)

    def retransform_energy(energy):
        return tf.convert_to_tensor(energy) * Config.particles_std + Config.particles_mean
        #return tf.convert_to_tensor(energy) * (Config.max_energy - Config.min_energy) + Config.min_energy
    '''
