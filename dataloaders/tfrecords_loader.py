import tensorflow as tf
import os


def dataloader():
    def decode(dataset, config):
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
            tf.reshape(parsed["image"], (config.img_width, config.img_height, 1)),
            tf.reshape(
                parsed["annotations"], (parsed["objects_no"], config.features_no)
            ),
        )

    def loader(config, stage="training"):
        files = config.tfrecords_files
        if stage == "valdiation":
            files = config.tfrecords_validation_files
        elif stage == "testing":
            files = config.tfrecords_test_files
        files = config.paths_to_global(files)
        dataset = tf.data.TFRecordDataset(
            files,
            buffer_size=config.tfrecords_buffer_size,
            num_parallel_reads=os.cpu_count(),
            compression_type=config.tfrecords_compression_type,  # "GZIP", "ZLIB", or ""
        ).map(lambda x: decode(x, config))
        return dataset

    return loader
