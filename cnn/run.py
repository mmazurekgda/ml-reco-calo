import numpy as np
#from tabulate import tabulate
import tensorflow as tf
import time
import pandas as pd
import logging as log
import argparse

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

# local
from cnn.config import Config
from cnn.dataset import Dataset
from cnn.core import YOLOCore


class CNN(YOLOCore):

    def __init__(self,
        model,
        dataloader,
        config=Config(),
        logger_level=log.CRITICAL,
    ):
        super().__init__(
            config=config,
            logger_level=logger_level,
        )

        if not model:
            log.error("Model not specified")
        self.model = model

        if not dataloader:
            log.error("Dataloader not specified")
        self.dataloader = dataloader

        # set default properties
        # self.batch_size = Config.batch_size
        # self.epochs = Config.epochs
        # self.validation_samples = Config.validation_samples
        # self.dataset_cache = Config.dataset_cache
        # self.early_stopping_patience = Config.early_stopping_patience
        # self.reduce_lr_patience = Config.reduce_lr_patience
        # self.reduce_lr_cooldown = Config.reduce_lr_cooldown
        # self.learning_rate = Config.learning_rate
        # self.anchors = Config.anchors
        # self.anchor_masks = Config.anchor_masks
        # self.backbone = Config.backbone
        # self.iou_ignore = Config.iou_ignore
        # self.load_weight_path = Config.load_weight_path
        # self.out_weight_path = Config.out_weight_path


    def transform_dataset(self, x, y):
        return (self.transform_images(x), self.transform_targets(y))

    def parse_dataset(self, dataset):
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.transform_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.dataset_cache:
            dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        return dataset

    """
    def infer(self):
        
       
        ts, first_file = Dataset().make_dataset()
        first_file_no = int(first_file.split('_')[2].split('.')[0])
        model = eval("model_{}".format(Config.backbone))()
        model.load_weights(Config.load_weight_path).expect_partial()
        for index, (x, y) in enumerate(ts.skip(0).take(Config.stats_events)):
            print("Evaluating {} ".format(index + first_file_no))
            img = x.numpy().reshape(Config.img_height, Config.img_width)
            input_img = tf.expand_dims(img, -1)
            input_img = tf.expand_dims(input_img, 0)
            input_img = CaloYolo.transform_images(input_img)

            y_test = CaloYolo.transform_targets(y)
            for y_t in y_test:
                print(y_t.shape, y_t[y_t[..., 0] != y_t[..., 1] ].shape)
                #print(y[y[...,0] > 0.0])
                #print(y_t[y_t[...,0] > 0.0])
            #print(y_test[0].shape)
            #y_test_0 = y_test[0][y_test[0][...,0] > 0.0]
            #y_test_1 = y_test[1][y_test[1][...,0] > 0.0]
            #y_test_2 = y_test[2][y_test[2][...,0] > 0.0]
            #print(y_test_0)
            #print(y_test_1)
            #print(y_test_2)

            out = model(input_img, training=False)
            clusters_df = pd.DataFrame.from_dict({
                'Energy Cluster YOLO': Config.retransform_energy(out[1]),
                'Cell X Min Cluster YOLO': out[0][..., 0] * Config.img_width,
                'Cell Y Min Cluster YOLO': out[0][..., 1] * Config.img_height,
                'Cell X Max Cluster YOLO': out[0][..., 2] * Config.img_width,
                'Cell Y Max Cluster YOLO': out[0][..., 3] * Config.img_height,
            })
            clusters_df.to_pickle('{}/event_{}_{}_yolo.pkl'.format(Config.conf_timestamp, Config.dataset_name, index + first_file_no))
            print(out[1].shape, y.shape)
        """

    def train(self):
        # Load data
        ds = self.dataset_loader()
        vs = self.dataset_loader()
        ds = self.parse_dataset(ds)
        vs = self.parse_dataset(vs)

        # Callbacks
        callbacks = []
        if self.reduce_lr:
            callbacks.append(
                ReduceLROnPlateau(
                    verbose=self.reduce_lr_verbosity,
                    patience=self.reduce_lr_patience,
                    cooldown=self.reduce_lr_cooldown,
                )
            )
        if early_stopping:
            callbacks.append(
                EarlyStopping(
                    patience=self.early_stopping_patience,
                    verbose=self.early_stopping_verbosity,
                    restore_best_weights=self.early_stopping_restore,
                )
            )
        if model_checkpoint:
            callbacks.append(
                ModelCheckpoint(
                    self.model_checkpoint_out_weight_path,
                    verbose=self.model_checkpoint_verbose,
                    save_weights_only=self.save_weights_only,
                    save_best_only=self.save_best_only,
                ),
            )
            #TensorBoard(log_dir=Config.tensorboard_dir,
            #            histogram_freq=1, batch_size=Config.batch_size, write_grads=True, write_graph=True)

        #with tf.distribute.MirroredStrategy().scope():
            #model = eval("model_{}".format(backbone))(training=True)
        log.debug(self.model.summary())
        if self.load_weight_path:
            self.model.load_weights(self.load_weight_path)

        optimizer = tf.keras.optimizers.Adam(
            lr=self.learning_rate
        )

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss()
        )

        # return the history 
        return self.model.fit(
            ds,
            epochs=self.epochs,
            steps_per_epoch=self.samples // self.batch_size,
            validation_data=vs,
            validation_steps=self.validation_samples,
            callbacks=callbacks,
        )


"""
if __name__ == "__main__":
    if not os.path.exists(Config.conf_timestamp):
        os.makedirs(Config.conf_timestamp)
    parsed_config = [[k.upper().replace('_', ' '), v if not callable(eval('Config.{}'.format(k))) else inspect.getsource(eval('Config.{}'.format(k)))] for k, v in Config.__dict__.items() if k[:1] != '_']
    print_n_dump('Training {}'.format(Config.training_time.strftime("%d-%m-%Y %H:%M:%S")))
    print_n_dump(tabulate(parsed_config))

    Normalizer.z_score_getter()
    # Train

    if Config.inference:
        CNN().infer()
    else:
        CNN().train()
"""
