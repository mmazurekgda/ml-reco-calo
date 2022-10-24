import numpy as np
import tensorflow as tf
import glob
import os
import math

from cnn.config import Config


class Dataset:
    def inference_dataset(self):
        test_tfrecords = glob.glob(Config.test_dataset_path)
        # this is specific to a particular format of a file
        test_tfrecords.sort(key=lambda e: int(e.split("_")[2].split(".")[0]))
        assert len(test_tfrecords) > 0
        dataset = tf.data.TFRecordDataset(
            test_tfrecords,
            buffer_size=10240,
            num_parallel_reads=tf.data.experimental.AUTOTUNE,
            compression_type=Config.compression_type,
        )
        return (
            dataset.map(
                lambda x: self.parse_image(x),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            ),
            test_tfrecords[0],
        )

    def training_dataset(self):
        train_tfrecords = glob.glob(Config.dataset_path)
        assert len(train_tfrecords) > 0
        dataset = tf.data.TFRecordDataset(
            train_tfrecords,
            buffer_size=10240,
            num_parallel_reads=tf.data.experimental.AUTOTUNE,
            compression_type=Config.compression_type,
        )
        dataset = dataset.map(
            lambda x: self.parse_image(x),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.take(Config.samples)
        if not Config.val_dataset_path:
            val_no = int(Config.val_dataset_split / 100.0 * Config.samples)
            val_dataset = dataset.take(val_no)
            dataset = dataset.skip(val_no)
        else:
            val_dataset = tf.data.TFRecordDataset(
                val_tfrecords,
                buffer_size=10240,
                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                compression_type=Config.compression_type,
            )
            val_dataset = val_dataset.map(
                lambda x: self.parse_image(x),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            assert len(val_tfrecords) > 0
        return (dataset, val_dataset)

    def parse_image(self, example_proto):
        data = tf.io.parse_single_example(example_proto, Config.dataset_feature)
        x = tf.cast(data["image/digits"], tf.float32) / Config.float_to_int
        x = Config.normalize_image(x)
        x = tf.reshape(
            x, (Config.img_height, Config.img_width)
        )  # (data['image/sizex'], data['image/sizey']))
        x = tf.expand_dims(x, -1)

        y_base = [
            data["image/particle/min_x"] / Config.img_width,
            data["image/particle/min_y"] / Config.img_height,
            data["image/particle/max_x"] / Config.img_width,
            data["image/particle/max_y"] / Config.img_height,
            Config.transform_energy(data["image/particle/energy"]),
            tf.cast(data["image/particle/pid"], tf.float32)
            / Config.classes_no,  # must be fixed
        ]

        y = tf.stack(y_base, axis=1)

        y = tf.cond(tf.size(y) == 6, lambda: tf.reshape(y, (1, 6)), lambda: y)

        # sort particles wrt to their energy so that those with higher will be more important in the grid
        y = tf.gather(y, tf.argsort(y[..., 4], name="DESCENDING"))

        # if tf.shape(y)[0] > 0:
        #    y = tf.cond(tf.size(y) == 6, lambda: tf.reshape(y, (1,  6)), lambda: y)
        #    y = tf.gather(y, tf.math.top_k(y[..., -2], k=tf.shape(y)[0]).indices)
        #
        # y = tf.cond(tf.size(y) == 6, lambda: tf.reshape(y, (1,  6)), lambda: y)

        # for i in tf.range(4):
        #    y = tf.gather(y, tf.squeeze(tf.where(y[..., i] >= 0.0)))
        #    y = tf.cond(tf.size(y) == 6, lambda: tf.reshape(y, (1,  6)), lambda: y)

        # tf.print(y.shape)

        # y = tf.gather(y, tf.squeeze(tf.where(y[..., 4] > Config.transform_energy(Config.energy_noise))))
        # y = tf.cond(tf.size(y) == 6, lambda: tf.reshape(y, (1,  6)), lambda: y)

        # y = tf.math.multiply(y, tf.pad(tf.reshape(y[..., -1] * 0 + Config.classes_no, (tf.shape(y)[0], 1)), [[0, 0], [tf.shape(y)[1] - 1, 0]], constant_values=1))
        y = tf.cast(y, tf.float32)
        return (x, y)
