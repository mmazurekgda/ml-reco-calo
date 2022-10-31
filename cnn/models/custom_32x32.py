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

class SimpleConv(Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        padding="same",
        batch_norm=True,
        kernel_regularizer=l2(0.0005),
        **kwargs
    ):

        super().__init__(
            filters,
            kernel_size,
            padding="same",
            use_bias=not batch_norm,
            kernel_regularizer=kernel_regularizer,
            **kwargs
        )
        self.batch_norm = batch_norm
        self.batch_norm_layer = BatchNormalization()

    def call(self, inputs):
        x = super().call(inputs)
        if self.batch_norm:
            x = self.batch_norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x

class Model(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.apply_refine = False
        self.apply_nms = False
        self.config = config
        self.conv_1 = SimpleConv(8, 3)
        self.conv_2 = SimpleConv(16, 3)
        self.conv_3 = SimpleConv(16, 3)
        self.conv_4 = SimpleConv(32, 3)
        self.conv_5 = SimpleConv(64, 3)
        self.conv_6 = SimpleConv(128, 3)
        self.conv_7 = SimpleConv(128, 1)
        self.conv_8 = SimpleConv(64, 1)
        self.conv_9 = SimpleConv(32, 1)
        self.conv_10 = SimpleConv(16, 1)
        self.conv_11 = SimpleConv(8, 1)
        self.conv_12 = SimpleConv(
            len(self.config.anchor_masks[0]) * (self.config.classes_no + 6),
            1,
            batch_norm=False,
        )
        self.lambda_1 = Lambda(
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
        )
        self.lambda_refine = Lambda(lambda x: self.refine(x))
        self.lambda_nms = Lambda(lambda x: self.nms(x))

    def call(
        self,
        inputs,
        training=True, # not doing the job because model(training) = model(valdiation)
    ):
        x = self._backbone(inputs)
        # below for testing only!
        if self.apply_refine:
            x = [self.lambda_refine(x_i) for x_i in x]
            if self.apply_nms:
                x = self.lambda_nms([x_i[:4] for x_i in x])
        return x

    def refine(self, x):
        return self.config.refine_boxes(x, self.config.anchors[self.config.anchor_masks[0]])

    def nms(self, x):
        return self.config.nms(x)

    def _backbone(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x_1 = self.conv_3(x)
        x = MaxPool2D(2, 2, "same")(x)
        # x = AveragePooling2D(2, 2, 'same')(x)
        x = x_2 = self.conv_4(x)
        x = MaxPool2D(2, 2, "same")(x)
        # x = AveragePooling2D(2, 2, 'same')(x)
        x = x_3 = self.conv_5(x)
        x = MaxPool2D(2, 2, "same")(x)
        # x = AveragePooling2D(2, 2, 'same')(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_3])
        x = self.conv_8(x)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_2])
        x = self.conv_9(x)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_1])
        x = self.conv_10(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        output_0 = self.lambda_1(x)
        return (output_0, )
