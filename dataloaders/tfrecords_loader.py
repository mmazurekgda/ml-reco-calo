import tensorflow as tf
import os


class DataLoader():
    def decode(self, dataset):
        parsed = tf.io.parse_single_example(
            dataset,
            {
                "image": tf.io.FixedLenSequenceFeature(
                    [], tf.float32, allow_missing=True
                ),
                "annotations": tf.io.FixedLenSequenceFeature(
                    [], tf.float32, allow_missing=True
                ),
                "objects_no": tf.io.FixedLenFeature([], dtype=tf.int64),
            },
        )
        return (
            tf.reshape(parsed["image"], (self.config.img_width, self.config.img_height, 1)),
            tf.reshape(
                parsed["annotations"], (parsed["objects_no"], self.config.input_features_no)
            ),
        )

    def __init__(self, config, stage="training"):
        self.config = config
        files = self.config.tfrecords_files
        if stage == "validation":
            files = self.config.tfrecords_validation_files
        elif stage == "testing":
            files = self.config.tfrecords_test_files
        files = self.config.paths_to_global(files)
        self.dataset = tf.data.TFRecordDataset(
            files,
            buffer_size=self.config.tfrecords_buffer_size,
            num_parallel_reads=os.cpu_count(),
            compression_type=self.config.tfrecords_compression_type,  # "GZIP", "ZLIB", or ""
        ).map(self.decode)
