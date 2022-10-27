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

registered_interruptions = 0


class StopTrainingSignal(Exception):
    pass


class CNNCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger=None):
        super().__init__()
        self.log = logger

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', 'UNKNOWN')
        val_loss = logs.get('val_loss', 'UNKNOWN')
        self.log.info(
            f"\n--> Epoch: {epoch}, 'loss': {loss}, 'val_loss': {val_loss}"
        )

    def on_batch_end(self, batch, logs=None):
        global registered_interruptions
        if registered_interruptions >= 2:
            self.model.stop_training = True


class CNN(CNNCore):
    def _stop_training_handler(self):
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
        self.model = model#(config)

        if not dataloader:
            self.log.error("-> Dataloader not specified")
        self.dataloader = dataloader(config)
        self.val_dataloader = dataloader(config, training=False)

        # Load data
        dataset = self.dataloader
        self.dataset = self.parse_dataset(dataset)
        val_dataset = self.val_dataloader
        self.val_dataset = self.parse_dataset(val_dataset)

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
        dataset = dataset.repeat()
        return dataset

    def train(self, no_devices=1):
        self.log.info("-> Preparing the training procedure...")
        # Callbacks
        callbacks = [CNNCustomCallback(logger=self.log)]
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

        self.model = self.model(self.config)

        if self.config.load_weight_path:
            paths = self.config.paths_to_global(paths)
            self.log.debug(f"-> Loading weights from: {paths}")
            self.model.load_weights(paths)

        self.model.summary(print_fn=lambda x: self.log.debug(x))

        self.log.debug("-> Adding the optimizer.")
        optimizer = tf.keras.optimizers.Adam(lr=self.config.learning_rate)

        self.log.debug("-> Compiling model...")
        self.model.compile(optimizer=optimizer, loss=self.loss())
        self.log.debug("-> Done.")
        self.log.info("-> Training...")
        try:
            signal.signal(signal.SIGINT, self._stop_training_handler())
            self.model.fit(
                self.dataset,
                epochs=self.config.epochs,
                validation_data=self.val_dataset,
                # TF 2.0: needed to avoid errors at the end of loop
                steps_per_epoch=self.config.samples // self.config.batch_size // no_devices,
                # validation_data=vs,
                # TF 2.0: needed to avoid errors at the end of loop
                validation_steps=self.config.validation_samples // self.config.batch_size // no_devices,
                callbacks=callbacks,
            )
        except StopTrainingSignal:
            self.log.fatal("Training stopped!")

        self.log.info("-> Training finished.")
        self.log.info("-> End of the training procedure.")
