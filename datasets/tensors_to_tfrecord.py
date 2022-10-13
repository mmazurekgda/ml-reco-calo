import tensorflow as tf
import logging as log
from tqdm import tqdm

def convert_tensors_to_tfrecord(
    tfrecord_file,
    iterator_to_dataset,
    samples,
    compression_type='ZLIB', # "GZIP", "ZLIB", or ""
    compression_level=9, # 0 to 9, or None.
    loss_compression_factor=None,
):
    logger = log.getLogger('MCRecoCalo')

    def float_list(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def int64_list(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    tf_file_options = tf.io.TFRecordOptions(
        compression_type=compression_type,
        compression_level=compression_level,
    )

    def range_verbose(iterator):
        if logger.level <= log.DEBUG:
            return tqdm(range(iterator))
        return range(iterator)

    with tf.io.TFRecordWriter(tfrecord_file, options=tf_file_options) as writer:
        logger.debug("Opened tfrecord_file: " + tfrecord_file)

        for i in range_verbose(samples):
            x, y = next(iterator_to_dataset)
            x = x.numpy()
            y = y.numpy()
            parser = float_list
            if loss_compression_factor:
                x = (x * loss_compression_factor).round().astype(int)
                y = (y * loss_compression_factor).round().astype(int)
                parser = int64_list

            feature = {
                'image': parser(x.reshape(-1)),
                'annotations': parser(y.reshape(-1)),
                'objects_no': int64_feature(y.shape[0])
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())
    logger.debug("Written to tfrecord_file: " + tfrecord_file)

