import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras import Model
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from cnn.config import Config
from cnn.backbone import DarknetConv
import sys
import logging as log


class CNNCore:
    def __init__(self, config=None):
        if not config:
            raise ValueError("No config passed!")
        self.config = config

        # FIXME: temporary workaround
        self.config.set_options(
            {
                "refine_boxes": self.refine_boxes,
                "nms": self.nms,
            },
            permit_when_frozen=True,
        )

        self.log = log.getLogger("MCRecoCalo")
        log.getLogger("tensorflow").setLevel(self.log.level)
        # self.config.classes_no = Config.classes_no
        # self.config.target_img_width = Config.target_img_width
        # self.config.target_img_height = Config.target_img_height
        # self.config.granularities = Config.granularities
        # LOSS
        # self.config.iou_ignore = Config.iou_ignore
        # NON MAXIMUM SUPPRESSION
        # self.config.iou_threshold = Config.iou_threshold
        # self.config.score_threshold = Config.score_threshold
        # self.config.max_boxes = Config.max_boxes
        # self.config.soft_nms_sigma = Config.soft_nms_sigma

    """
    def output(self, filters, anchors_tmp, name=None):
        def yolo_output(x_in):
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters * 2, 3)
            x_boxes = DarknetConv(x, anchors_tmp * (self.config.classes_no + 5), 1, batch_norm=False)
            x_boxes = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors_tmp, (self.config.classes_no + 5))))(x_boxes)
            x_boxes_left, x_boxes_right = Lambda(lambda x: tf.split(x, (4, 1 + self.config.classes_no), axis=-1))(x_boxes)
            x_energy = Dense(anchors_tmp * 1)(x)
            x_energy = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors_tmp, 1)))(x_energy)
            x = Lambda(lambda x: tf.concat([x[0], x[2], x[1]], -1))([x_boxes_left, x_boxes_right, x_energy])
            return tf.keras.Model(inputs, x, name=name)(x_in)
        return yolo_output
    """

    def refine_boxes(self, pred, anchors):
        # YOLO's anchor boxes refinement
        # pred: (batch_size, grid_y, grid_x, anchors, (x, y, w, h, energy, obj, ...classes_no))
        grid_size_y = tf.shape(pred)[1]
        grid_size_x = tf.shape(pred)[2]
        box_xy, box_wh, energy, objectness, class_probs = tf.split(
            pred, (2, 2, 1, 1, self.config.classes_no), axis=-1
        )

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        # energy = tf.tanh(energy)
        #     class_probs = tf.sigmoid(class_probs)
        class_probs = tf.sigmoid(class_probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = tf.meshgrid(tf.range(grid_size_x), tf.range(grid_size_y))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(
            [grid_size_x, grid_size_y], tf.float32
        )
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, energy, objectness, class_probs, pred_box

    def nms(self, outputs):
        # YOLO Non - maximum suppression
        # boxes, conf, type
        b, e, c, t = [], [], [], []

        for o in outputs:
            b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
            e.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
            c.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
            t.append(tf.reshape(o[3], (tf.shape(o[3])[0], -1, tf.shape(o[3])[-1])))

        bbox = tf.concat(b, axis=1)
        energy = tf.concat(e, axis=1)
        confidence = tf.concat(c, axis=1)
        class_probs = tf.concat(t, axis=1)

        scores = confidence  # * class_probs

        boxes = tf.reshape(bbox, (-1, 4))
        scores = tf.reshape(scores, [-1])
        energy = tf.reshape(energy, [-1])
        class_probs = tf.reshape(class_probs, (-1, self.config.classes_no))
        # classes_no = class_probs

        # above_thr = tf.squeeze(tf.where(tf.greater_equal(scores, 0.5)))
        # boxes = tf.gather(boxes, above_thr)
        # energy = tf.gather(energy, above_thr)
        # classes_no = tf.gather(classes_no, above_thr)
        # scores = tf.gather(scores, above_thr)

        # boxes, scores, classes_no, valid_detections = tf.image.combined_non_max_suppression(
        #     boxes=boxes,
        #     scores=scores,
        #     max_output_size_per_class=g,
        #     max_total_size=max_boxes,
        #     iou_threshold=iou_threshold,
        #     score_threshold=score_threshold
        # )

        # boxes = tf.gather(boxes, indices=indices)
        # energy = tf.gather(energy, indices=indices)
        # scores = tf.gather(scores, indices=indices)
        indices, scores = tf.image.non_max_suppression_with_scores(
            boxes=boxes,
            scores=scores,
            # max_output_size_per_class=max_boxes,
            max_output_size=self.config.max_boxes,
            iou_threshold=self.config.iou_threshold,
            score_threshold=self.config.score_threshold,
            soft_nms_sigma=self.config.soft_nms_sigma,
        )

        boxes = tf.gather(boxes, indices=indices)
        energy = tf.gather(energy, indices=indices)
        # scores = tf.gather(scores, indices=indices)
        classes_no = tf.gather(class_probs, indices=indices)

        return boxes, energy, classes_no, scores

    def broadcast_iou(self, box_1, box_2):
        # box_1: (..., (x1, y1, x2, y2)) prediction box
        # box_2: (N, (x1, y1, x2, y2)) true box

        # broadcast boxes
        box_1 = tf.expand_dims(box_1, -2)
        box_2 = tf.expand_dims(box_2, 0)
        # new_shape: (..., N, (x1, y1, x2, y2))
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
        box_1 = tf.broadcast_to(box_1, new_shape)
        box_2 = tf.broadcast_to(box_2, new_shape)

        # get width and height of the intersection of a true box and prediction box
        int_w = tf.maximum(
            tf.minimum(box_1[..., 2], box_2[..., 2])
            - tf.maximum(box_1[..., 0], box_2[..., 0]),
            0,
        )
        int_h = tf.maximum(
            tf.minimum(box_1[..., 3], box_2[..., 3])
            - tf.maximum(box_1[..., 1], box_2[..., 1]),
            0,
        )
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

        # intersection over union IoU = I / (A + B - I)
        return int_area / (box_1_area + box_2_area - int_area)

    def yolo_loss(self, anchors):
        def yolo_loss(y_true, y_pred):
            # 1. transform all pred outputs
            # y_pred: (batch_size, grid_x, grid_y, anchors, (x, y, w, h, energy, obj, ...cls))
            pred_box, pred_energy, pred_obj, pred_class, pred_xywh = self.refine_boxes(
                y_pred, anchors
            )
            pred_xy = pred_xywh[..., 0:2]
            pred_wh = pred_xywh[..., 2:4]

            # 2. transform all true outputs
            # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, energy, obj, cls))
            # split y_true into tensors of shapes: 4, 1, 1, 1 along the last axis ->
            # true_box =        batch_size x grid x grid x anchors x        (x1, y1, x2, y2)
            # true_energy =     batch_size x grid x grid x anchors x        energy
            # true_obj =        batch_size x grid x grid x anchors x        obj
            # true_class_idx =  batch_size x grid x grid x anchors x        cls
            true_box, true_energy, true_obj, true_class_idx = tf.split(
                y_true, (4, 1, 1, 1), axis=-1
            )

            # get center of the boxes
            # true_xy =         batch_size x grid x grid x anchors x        (x_center, y_center)
            true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2

            # impose penalty for not being in

            # get width and height of the boxes
            # true_wh =         batch_size x grid x grid x anchors x        (width, height)
            true_wh = true_box[..., 2:4] - true_box[..., 0:2]

            # give higher weights to small boxes
            # i.e.
            # max box would be width * height = 1 so box_loss_scale = 1
            # small box would be ex. width * height = 0.01 so box_loss_scale = 1.99 (bigger)
            box_loss_scale = 1.0  # 2 - true_wh[..., 0] * true_wh[..., 1]
            energy_loss_scale = 1.0  # ((true_energy[..., 0] * Config.z_score_std + Config.z_score_mean) / 4000.0 / 7.66 + 1.0) ** 2.

            # 3. inverting the pred box equations
            grid_size_x = tf.shape(y_true)[2]
            grid_size_y = tf.shape(y_true)[1]

            # create a grid: grid_size x grid_size
            grid = tf.meshgrid(tf.range(grid_size_x), tf.range(grid_size_y))

            grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
            true_xy = true_xy * tf.cast(
                [grid_size_x, grid_size_y], tf.float32
            ) - tf.cast(grid, tf.float32)
            true_wh = tf.math.log(true_wh / anchors)
            true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

            # 4. calculate all masks
            # squeeze removes specified dimensions of size 1 (here the last dim)
            obj_mask = tf.squeeze(true_obj, -1)
            # ignore false positive when iou is over threshold
            # boolean_mask removes all values from true_box that are specified as False in a mask
            # (here obj_mask that was cast to bool)
            true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))

            # computes max iou across all dimensions
            best_iou = tf.reduce_max(
                self.broadcast_iou(pred_box, true_box_flat), axis=-1
            )
            # ignore_mask =  iou < threshold ? 0.0 : 1.0
            # TODO: try running YOLO on ignore thresholds bigger than 0.5
            ignore_mask = tf.cast(best_iou < self.config.iou_ignore, tf.float32)

            # 5. calculate all losses
            # objectness x choose_smaller_boxes x
            xy_loss = (
                obj_mask
                * box_loss_scale
                * energy_loss_scale
                * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            )
            wh_loss = (
                obj_mask
                * box_loss_scale
                * energy_loss_scale
                * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
            )
            energy_loss = (
                obj_mask
                * box_loss_scale
                * energy_loss_scale
                * tf.reduce_sum(tf.square(true_energy - pred_energy), axis=-1)
            )
            # energy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square((tf.math.log(K.clip(true_energy, K.epsilon(), None) + 1.0) - tf.math.log(K.clip(pred_energy, K.epsilon(), None) + 1.0)) / tf.math.log(10.0)), axis=-1)
            obj_loss = binary_crossentropy(true_obj, pred_obj)
            obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

            class_loss = obj_mask * sparse_categorical_crossentropy(
                true_class_idx, pred_class
            )

            # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
            xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
            wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
            energy_loss = tf.reduce_sum(energy_loss, axis=(1, 2, 3))
            obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
            class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

            return (
                xy_loss + obj_loss + wh_loss + class_loss
            )  # + .25 * energy_loss  #+ 0.125 * class_loss

        return yolo_loss

    def loss(self):
        return [
            self.yolo_loss(self.config.anchors[mask])
            for mask in self.config.anchor_masks
        ]

    def transform_targets_for_output(
        self, y_true, grid_size_x, grid_size_y, anchor_idxs
    ):
        # y_true: (boxes, (x1, y1, x2, y2, energy, class, best_anchor))
        # remember! image = height(rows) x width(cols) and output x -> width and y ->height
        # y_true_out_l:  (grid_y, grid_x, anchors, [x, y, w, h,])
        y_true_out_l = tf.zeros((grid_size_y, grid_size_x, tf.shape(anchor_idxs)[0], 4))
        # y_true_out_r:  (grid_y, grid_x, anchors, [obj, class,])
        y_true_out_r = tf.zeros((grid_size_y, grid_size_x, tf.shape(anchor_idxs)[0], 2))
        # y_true_out_en: (grid_y, grid_x, anchors, [energy,])
        y_true_out_en = tf.zeros(
            (grid_size_y, grid_size_x, tf.shape(anchor_idxs)[0], 1)
        )
        min_anchor_idx = tf.cast(anchor_idxs[0], tf.int32)
        y_anchored = y_true
        # slice propsoal s.t. only those with anchor in the particular yolo sungrid are taken int account

        ys_selected_anchors = [
            tf.equal(int(anchor_idxs[idx]), tf.cast(y_true[..., 6], tf.int32))
            for idx in range(len(anchor_idxs))
        ]
        ys_selected_anchor = ys_selected_anchors[0]
        if tf.shape(anchor_idxs)[0] > 1:
            for anchor_idx in anchor_idxs[1:]:
                ys_selected_anchor = tf.logical_or(
                    ys_selected_anchor,
                    tf.equal(int(anchor_idx), tf.cast(y_true[..., 6], tf.int32)),
                )
        y_anchored = tf.gather(
            y_true, tf.where(tf.squeeze(ys_selected_anchor, axis=0)), axis=1
        )
        box_xy = (y_anchored[..., 0:2] + y_anchored[..., 2:4]) / 2
        # find yolo's subgrid coordinates
        grid_x = tf.cast(box_xy[..., 0:1] // (1 / grid_size_x), tf.int32)
        grid_y = tf.cast(box_xy[..., 1:2] // (1 / grid_size_y), tf.int32)
        # fill objectness with ones
        obj = tf.ones(tf.shape(y_anchored[..., 0:1]))
        # prepare values for the yolo subgrid
        updates_l = tf.squeeze(y_anchored[..., 0:4])
        updates_l = tf.cond(
            tf.size(updates_l) == 4,
            lambda: tf.reshape(updates_l, (1, 4)),
            lambda: updates_l,
        )
        updates_r = tf.squeeze(tf.concat([obj, y_anchored[..., 5:6]], axis=-1))
        updates_r = tf.cond(
            tf.size(updates_r) == 2,
            lambda: tf.reshape(updates_r, (1, 2)),
            lambda: updates_r,
        )
        updates_en = tf.squeeze(y_anchored[..., 4:5], axis=[0, 2])
        # tf.print(tf.shape(updates_en))
        # tf.print(updates_en)
        updates_en = tf.cond(
            tf.size(updates_en) == 1,
            lambda: tf.reshape(updates_en, (1, 1)),
            lambda: updates_en,
        )
        # prepare indices that will be use to put values in the yolo grid
        indices = tf.squeeze(
            tf.concat(
                [
                    grid_y,
                    grid_x,
                    tf.cast(y_anchored[..., 6:7], tf.int32) - min_anchor_idx,
                ],
                axis=-1,
            )
        )
        indices = tf.cond(
            tf.size(indices) == 3, lambda: tf.reshape(indices, (1, 3)), lambda: indices
        )
        # scatter and update/sum
        y_true_out_l = tf.tensor_scatter_nd_update(y_true_out_l, indices, updates_l)
        y_true_out_r = tf.tensor_scatter_nd_update(y_true_out_r, indices, updates_r)
        y_true_out_en = tf.tensor_scatter_nd_add(y_true_out_en, indices, updates_en)
        y_true_out = tf.concat(
            [
                y_true_out_l,
                y_true_out_en,
                y_true_out_r,
            ],
            axis=-1,
        )
        return y_true_out

    def transform_targets(self, y_train):
        y_outs = []
        grid_sizes = [
            (
                self.config.target_img_width // granularity_x,
                self.config.target_img_height // granularity_y,
            )
            for granularity_x, granularity_y in self.config.granularities
        ]

        # calculate anchor index for true boxes
        # y_train = tf.sparse.to_dense(y_train.to_sparse())
        y_train = tf.expand_dims(y_train, axis=0)
        anchors_tf = tf.cast(self.config.anchors, tf.float32)
        anchor_area = anchors_tf[..., 0] * anchors_tf[..., 1]  # 10x13= 130
        box_wh = y_train[..., 2:4] - y_train[..., 0:2]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors_tf)[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = tf.minimum(box_wh[..., 0], anchors_tf[..., 0]) * tf.minimum(
            box_wh[..., 1], anchors_tf[..., 1]
        )
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)
        y_train = tf.concat([y_train, anchor_idx], axis=-1)

        for anchor_idxs, (grid_size_x, grid_size_y) in zip(
            self.config.anchor_masks, grid_sizes
        ):
            y_outs.append(
                self.transform_targets_for_output(
                    y_train, grid_size_x, grid_size_y, anchor_idxs
                )
            )

        return tuple(y_outs)

    def transform_images(self, x_train):
        return tf.image.resize(
            x_train, (self.config.target_img_height, self.config.target_img_width)
        )
