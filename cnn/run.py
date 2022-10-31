import numpy as np

# from tabulate import tabulate
import tensorflow as tf
import time
import logging as log
import argparse
import signal

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

# local
from cnn.config import Config
from cnn.dataset import Dataset
from cnn.core import CNNCore
from cnn.callbacks import (
    StopTrainingSignal,
    CNNLoggingCallback,
    CNNTestingCallback,
)


class CNN(CNNCore):
    def __init__(
        self,
        model=None,
        dataloader=None,
        config=None,
    ):
        super().__init__(
            config=config,
        )

        if not model:
            self.log.error("-> Model not specified")
        self.model = model  # (config)

        if not dataloader:
            self.log.error("-> Dataloader not specified")
        self.dataloader = dataloader(config)
        self.val_dataloader = dataloader(config, stage="validation")
        self.test_dataloader = dataloader(config, stage="testing")

        # Load data
        dataset = self.dataloader
        self.dataset = self.parse_dataset(dataset)
        val_dataset = self.val_dataloader
        self.val_dataset = self.parse_dataset(val_dataset)
        # FIXME: idiotic!
        self.test_dataset = self.test_dataloader

    def transform_dataset(self, x, y):
        return (self.transform_images(x), self.transform_targets(y))

    def parse_dataset(self, dataset):
        dataset = dataset.map(
            self.transform_dataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.batch(self.config.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )
        if self.config.dataset_cache:
            dataset = dataset.cache()
        # if self.config.shuffle:
        #    dataset = dataset.shuffle()
        if self.config.dataset_repeat:
            dataset = dataset.repeat()
        return dataset

    def train(self, no_devices=1):
        self.log.info("-> Preparing the training procedure...")

        # Callbacks
        callbacks = []

        standard_log_callback = CNNLoggingCallback(logger=self.log)
        callbacks.append(standard_log_callback)

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
        if self.config.tensorboard:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.config.tensorboard_log_dir,
                    histogram_freq=self.config.tensorboard_histogram_freq,
                    write_graph=self.config.tensorboard_write_graph,
                    write_images=self.config.tensorboard_write_images,
                    # incompatible with TF 2.0
                    # write_steps_per_second=self.config.tensorboard_write_steps_per_second,
                    update_freq=self.config.tensorboard_update_freq,
                    profile_batch=self.config.tensorboard_profile_batch,
                    embeddings_freq=self.config.tensorboard_embeddings_freq,
                    embeddings_metadata=self.config.tensorboard_embeddings_metadata,
                )
            )
        if self.config.testing:
            callbacks.append(
                CNNTestingCallback(
                    self.config,
                    self.log,
                    self.test_dataset,
                    self.transform_images,
                )
            )

        self.model = self.model(self.config)

        if self.config.load_weight_path:
            paths = self.config.paths_to_global(paths)
            self.log.debug(f"-> Loading weights from: {paths}")
            self.model.load_weights(paths)

        input_shape = [self.config.batch_size, None, None, self.config.channels]
        # self.model(tf.ones(shape=input_shape))
        self.model.build(input_shape)
        self.model.summary(print_fn=lambda x: self.log.debug(x))

        self.log.debug("-> Adding the optimizer.")
        optimizer = tf.keras.optimizers.Adam(lr=self.config.learning_rate)

        self.log.debug("-> Compiling model...")
        self.model.compile(optimizer=optimizer, loss=self.loss())
        self.log.debug("-> Done.")
        self.log.info("-> Training...")
        try:
            signal.signal(signal.SIGINT, standard_log_callback.stop_training_handler())
            self.model.fit(
                self.dataset,
                epochs=self.config.epochs,
                validation_data=self.val_dataset,
                # TF 2.0: needed to avoid errors at the end of loop
                steps_per_epoch=self.config.samples
                // self.config.batch_size
                // no_devices,
                # validation_data=vs,
                # TF 2.0: needed to avoid errors at the end of loop
                validation_steps=self.config.validation_samples
                // self.config.batch_size
                // no_devices,
                callbacks=callbacks,
            )
        except StopTrainingSignal:
            self.log.fatal("Training stopped!")

        self.log.info("-> Training finished.")
        self.log.info("-> End of the training procedure.")
