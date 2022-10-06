import numpy as np
import time
import tensorflow as tf
import math
import os

class Config:
    training_time = time
    conf_timestamp = training_time.strftime("%Y%m%d_%H%M%S")

    # TYPE
    inference = False

    # BACKBONE
    backbone = 'tiny'

    # DATASET
    val_dataset_path = None #'parsed_bkstargamma/*.tf'
    test_dataset_path = '*.tf' #'parsed_bkstargamma/*.tf'
    val_dataset_split = 10  # percent
    compression_type = 'GZIP'

    # LOADABLE OPTIONS (WEIGHTS etc.)
    out_weight_path = "{}/weights.tf".format(conf_timestamp)
    options_dir = None
    load_weight_path = "{}/weights.tf".format(options_dir) if options_dir else None
    params_path = "{}/params.json".format(options_dir) if options_dir else None
    digits_mean = None
    digits_std = None
    digits_max = None
    digits_min = None
    particles_mean = None
    particles_std = None
    particles_max = None
    particles_min = 0.

    # TRAINING OPTIONS
    learning_rate = 1e-4
    epochs = 30
    batch_size = 1
    samples = 100
    validation_samples = 100
    shuffle_buffer_size = 1000  # None = do not shuffle
    tensorboard_dir = '{}/logs'.format(conf_timestamp)
    early_stopping_patience = 6
    reduce_lr_patience = 5
    reduce_lr_cooldown = 5
    # !!!
    iou_ignore = .5
    dataset_cache = False

    ### INFERENCE OPTIONS ###
    max_boxes = 1500
    classes = ['one']
    iou_threshold = .5
    score_threshold = .5
    soft_nms_sigma = .5
    stats_events = 1

    ### INPUT (image) OPTIONS ###
    float_to_int = 10000.0
    img_width = 384
    img_height = 312
    target_img_width = 384
    target_img_height = 384
    channels = 1
    upscale = 1

    ### OUTPUT (particles) OPTIONS ###
    anchors_not_parsed = np.array([(6, 6), (9, 9), (18, 18)], np.float32)
    anchor_masks = np.array([[2], [1], [0]])
    granularities = [(8, 8), (4, 4), (2, 2)]  # ( width x height)  # yolo [32, 16, 8] and yol3 [16, 8, 4, 2]
    #granularities = [(32, 32), (16, 16), (8, 8)]  # ( width x height)  # yolo [32, 16, 8] and yol3 [16, 8, 4, 2]

    @staticmethod
    def normalize_image(x):
        return (x - Config.digits_mean) / Config.digits_std
        #return tf.where(tf.less(x - Config.min_digit, 1.0), 0.0, ((x - Config.min_digit) / math.log(10.0)) / (Config.max_digit - Config.min_digit))
        #return tf.where(tf.less(x - Config.min_digit, 0.0), 0.0, ((x - Config.min_digit) / (Config.max_digit - Config.min_digit)))
        #return tf.where(tf.less(x, Config.min_digit), 1.0,  tf.math.log(x - Config.min_digit + 1.) / math.log(10.) / math.log10(Config.max_digit - Config.min_digit + 1.))
        #return tf.where(tf.less(x, Config.min_digit), 0.0,  tf.math.log(x - Config.min_digit + 1.) / math.log(10.) / math.log10(Config.max_digit - Config.min_digit + 1.))

    @staticmethod
    def transform_energy(energy):
        return (tf.convert_to_tensor(energy) - Config.particles_mean) / Config.particles_std
        #return (tf.convert_to_tensor(energy) - Config.min_energy) / (Config.max_energy - Config.min_energy)

    @staticmethod
    def retransform_energy(energy):
        return tf.convert_to_tensor(energy) * Config.particles_std + Config.particles_mean
        #return tf.convert_to_tensor(energy) * (Config.max_energy - Config.min_energy) + Config.min_energy


    ### DO NOT CONFIGURE THIS PART ###
    anchors = anchors_not_parsed / [img_width, img_height]
    classes_no = len(classes) if isinstance(classes, list) else 1

    dataset_feature = {
        'image/event_index': tf.io.FixedLenFeature([], tf.int64),
        'image/digits': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'image/particle/pid': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'image/particle/min_x': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'image/particle/min_y': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'image/particle/max_x': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'image/particle/max_y': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'image/particle/energy': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }

