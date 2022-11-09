import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    MaxPool2D,
    Lambda,
    AveragePooling2D,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras.regularizers import l2


def SimpleConv(x, filters, kernel_size, batch_norm=True, **kwargs):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=not batch_norm,
        kernel_regularizer=l2(0.0005),
        **kwargs
    )(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


class Model:
    def __init__(self, config):
        self.config = config
        self.apply_refine = False
        self.apply_nms = False
        self.model = self.get_model()

    def get_model(self, training=True):
        x = inputs = tf.keras.Input([None, None, self.config.channels], name="input")
        x = SimpleConv(x, 8, 3)
        x = SimpleConv(x, 16, 3)
        x = x_1 = SimpleConv(x, 16, 3)
        x = MaxPool2D(2, 2, "same")(x)
        # x = AveragePooling2D(2, 2, 'same')(x)
        x = x_2 = SimpleConv(x, 32, 3)
        x = MaxPool2D(2, 2, "same")(x)
        # x = AveragePooling2D(2, 2, 'same')(x)
        x = x_3 = SimpleConv(x, 64, 3)
        x = MaxPool2D(2, 2, "same")(x)
        # x = AveragePooling2D(2, 2, 'same')(x)
        x = SimpleConv(x, 128, 3)
        x = SimpleConv(x, 128, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_3])
        x = SimpleConv(x, 64, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_2])
        x = SimpleConv(x, 32, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_1])
        x = SimpleConv(x, 16, 1)
        x = SimpleConv(x, 8, 1)

        x = SimpleConv(
            x,
            len(self.config.anchor_masks[0]) * (self.config.classes_no + 6),
            1,
            batch_norm=False,
        )
        output_0 = Lambda(
            lambda x: tf.reshape(
                x,
                (
                    -1,
                    tf.shape(x)[1],
                    tf.shape(x)[2],
                    len(self.config.anchor_masks[0]),
                    (self.config.classes_no + 6),
                ),
            )
        )(x)
        """
        if not training or self.apply_refine or self.apply_nms:
            boxes_0 = Lambda(
                lambda x: self.config.refine_boxes(x, self.config.anchors[self.config.anchor_masks[0]]),
                name="simple_boxes_0",
            )(output_0)
            if training and not self.apply_nms:
                return tf.keras.Model(inputs, (boxes_0,), name="simple")
            else:
                outputs = Lambda(lambda x: self.config.nms(x), name="yolo_nms")((boxes_0[:4],))
                return tf.keras.Model(inputs, outputs, name="simple_model")
        """
        return tf.keras.Model(inputs, (output_0,), name="simple")

    def __getattr__(self, name):
        return getattr(self.model, name)
