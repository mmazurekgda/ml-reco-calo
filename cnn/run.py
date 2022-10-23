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
from cnn.core import CNNCore


class CNN(CNNCore):

    def __init__(self,
        model,
        dataloader,
        config=Config(),
    ):
        super().__init__(
            config=config,
        )

        if not model:
            self.log.error("Model not specified")
        self.model = model(config)

        if not dataloader:
            self.log.error("Dataloader not specified")
        self.dataloader = dataloader(config)
        self.val_dataloader = dataloader(config, training=False)

        if self.config.load_weight_path:
            self.model.load_weights(self.config.load_weight_path)

    def transform_dataset(self, x, y):
        return (
            self.transform_images(x),
            self.transform_targets(y)
        )

    def parse_dataset(self, dataset):
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.map(
            self.transform_dataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.batch(self.config.batch_size)
        if self.config.dataset_cache:
            dataset = dataset.cache()
        return dataset

    def train(self):
        # Load data
        dataset = self.dataloader
        dataset = self.parse_dataset(dataset)
        val_dataset = self.val_dataloader
        val_dataset = self.parse_dataset(val_dataset)

        # Callbacks
        callbacks = []
        if self.config.reduce_lr:
            callbacks.append(
                ReduceLROnPlateau(
                    verbose=self.config.reduce_lr_verbosity,
                    patience=self.config.reduce_lr_patience,
                    cooldown=self.config.reduce_lr_cooldown,
                )
            )
        if self.config.early_stopping:
            callbacks.append(
                EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    verbose=self.config.early_stopping_verbosity,
                    restore_best_weights=self.config.early_stopping_restore,
                )
            )
        if self.config.model_checkpoint:
            callbacks.append(
                ModelCheckpoint(
                    self.config.model_checkpoint_out_weight_path,
                    verbose=self.config.model_checkpoint_verbosity,
                    save_weights_only=self.config.model_checkpoint_save_weights_only,
                    save_best_only=self.config.model_checkpoint_save_best_only,
                ),
            )
            #TensorBoard(log_dir=Config.tensorboard_dir,
            #            histogram_freq=1, batch_size=Config.batch_size, write_grads=True, write_graph=True)

        #with tf.distribute.MirroredStrategy().scope():
            #model = eval("model_{}".format(backbone))(training=True)
        self.log.debug(self.model.summary())

        optimizer = tf.keras.optimizers.Adam(
            lr=self.config.learning_rate
        )

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss()
        )

        # return the history 
        return self.model.fit(
            dataset,
            epochs=self.config.epochs,
            validation_data=val_dataset,
            # steps_per_epoch=self.config.samples // self.config.batch_size,
            # validation_data=vs,
            # validation_steps=self.config.validation_samples,
            callbacks=callbacks,
        )

