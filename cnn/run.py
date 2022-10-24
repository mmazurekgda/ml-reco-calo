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
        model=None,
        dataloader=None,
        config=None,
    ):
        super().__init__(
            config=config,
        )

        if not model:
            self.log.error("-> Model not specified")
        self.model = model(config)

        if not dataloader:
            self.log.error("-> Dataloader not specified")
        self.dataloader = dataloader(config)
        self.val_dataloader = dataloader(config, training=False)

        if self.config.load_weight_path:
            paths = self.config.paths_to_global(paths)
            self.log.debug(f"-> Loading weights from: {paths}")
            self.model.load_weights(paths)

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
            self.log.debug("-> Adding ReduceLROnPlateau callback")
            callbacks.append(
                ReduceLROnPlateau(
                    verbose=self.config.reduce_lr_verbosity,
                    patience=self.config.reduce_lr_patience,
                    cooldown=self.config.reduce_lr_cooldown,
                )
            )
        if self.config.early_stopping:
            self.log.debug("-> Adding EarlyStopping callback")
            callbacks.append(
                EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    verbose=self.config.early_stopping_verbosity,
                    restore_best_weights=self.config.early_stopping_restore,
                )
            )
        if self.config.model_checkpoint:
            self.log.debug("-> Adding ModelCheckpoint callback")
            callbacks.append(
                ModelCheckpoint(
                    self.config.model_checkpoint_out_weight_file,
                    verbose=self.config.model_checkpoint_verbosity,
                    save_weights_only=self.config.model_checkpoint_save_weights_only,
                    save_best_only=self.config.model_checkpoint_save_best_only,
                ),
            )
            #TensorBoard(log_dir=Config.tensorboard_dir,
            #            histogram_freq=1, batch_size=Config.batch_size, write_grads=True, write_graph=True)

        #with tf.distribute.MirroredStrategy().scope():
            #model = eval("model_{}".format(backbone))(training=True)
        self.model.summary(print_fn=lambda x: self.log.debug(x))

        self.log.debug("-> Adding the optimizer.")
        optimizer = tf.keras.optimizers.Adam(
            lr=self.config.learning_rate
        )

        self.log.debug("-> Compiling model...")
        self.model.compile(
            optimizer=optimizer,
            loss=self.loss()
        )
        self.log.debug("-> Done.")

        self.log.info("-> Training...")
        self.model.fit(
            dataset,
            epochs=self.config.epochs,
            validation_data=val_dataset,
            # steps_per_epoch=self.config.samples // self.config.batch_size,
            # validation_data=vs,
            # validation_steps=self.config.validation_samples,
            callbacks=callbacks,
        )
        self.log.info("-> Training finished.")

