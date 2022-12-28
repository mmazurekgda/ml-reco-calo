import numpy as np
import time
from collections import defaultdict
import tensorflow as tf
from scipy.optimize import linear_sum_assignment


def flatten(S):
    if len(S) == 0:
        return S
    arr = []
    for elem in S:
        if isinstance(elem, list):
            arr += np.array(elem).flatten().tolist()
        elif isinstance(elem, np.ndarray):
            arr += elem.flatten().tolist()
        else:
            arr.append(elem)
    return np.array(arr)


def ragged_to_normal(array):
    # FIXME: this is a bit redundant, but ragged data is tricky...
    if tf.is_tensor(array):
        return flatten(array.numpy())
    elif isinstance(array, tf.RaggedTensor):
        # TF 2.0 RaggedTensor does not support .numpy()
        return flatten(array.to_list())
    elif isinstance(array, np.ndarray):
        return flatten(array)
    else:
        raise NotImplementedError()


def convert_data(config, tests):
    tests["pred_energy"] = config.convert_to_energy(tests["pred_energy"])
    tests["true_energy"] = config.convert_to_energy(tests["true_energy"])
    for position_type in ["pred_position", "true_position"]:
        xmin = config.convert_to_position(tests[position_type][..., 0:1], dim="x")
        ymin = config.convert_to_position(tests[position_type][..., 1:2], dim="y")
        xmax = config.convert_to_position(tests[position_type][..., 2:3], dim="x")
        ymax = config.convert_to_position(tests[position_type][..., 3:4], dim="y")
        if isinstance(tests[position_type], tf.RaggedTensor):
            tests[position_type] = tf.concat([xmin, ymin, xmax, ymax], -1)
        else:
            tests[position_type] = np.concatenate([xmin, ymin, xmax, ymax], -1)

    tests["images"] = config.convert_to_hit_energy(tests["images"])


def get_cost_matrix(A, B, ndim=3):
    a = A[..., 0:ndim]
    b = B[..., 0:ndim]
    na = np.sum(np.square(a), axis=-1)
    nb = np.sum(np.square(b), axis=-1)
    na = np.expand_dims(na, axis=-1)
    nb = np.expand_dims(nb, axis=-2)
    return na - 2 * np.matmul(a, b.T) + nb


def linear_sum_assignment_with_inf(cost_matrix):
    # taken from
    # https://github.com/scipy/scipy/issues/6900#issuecomment-451735634
    # this is needed to account for the fact that some of the links
    # in the bipartite graph are more favorable
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder
    return linear_sum_assignment(cost_matrix)


def particle_matching(
    A,  # -> A must be true!
    B,  # -> B must be predicted!
    max_dist_diff=None,
    max_energy_diff=None,
    energy_in_log=True,
):

    banned_id_pairs = np.array([])
    # flag those relations with energy cost > max_energy_diff
    if max_energy_diff:
        A_e = A[..., 2:3].astype(np.float32)
        B_e = B[..., 2:3].astype(np.float32)
        if energy_in_log:
            # workaround: now some values in the predicted array might be <= 0
            # we set this to some small epsilon, so that if it is relatively far away
            # from the true values, which must be > 0 (energy)
            A_e[A_e <= 0.0] = 1e-30
            B_e[B_e <= 0.0] = 1e-30
            # if (A_e <= 0.0).any() or (B_e <= 0.0).any():
            #     raise ValueError("Energy must be >= 0.")
            A_e = np.log10(A_e)
            B_e = np.log10(B_e)
        cost_e = np.sqrt(get_cost_matrix(A_e, B_e, ndim=1).astype(np.float32))
        banned_id_pairs_e = np.argwhere(cost_e >= max_energy_diff)
        if not banned_id_pairs.any():
            banned_id_pairs = banned_id_pairs_e
        else:
            banned_id_pairs = np.concatenate(
                [
                    banned_id_pairs,
                    banned_id_pairs_e,
                ]
            )

    # flag those relations with distance > max_dist_diff
    if max_dist_diff:
        A_d = A[..., 0:2].astype(np.float32)
        B_d = B[..., 0:2].astype(np.float32)
        cost_d = np.sqrt(get_cost_matrix(A_d, B_d, ndim=2))
        banned_id_pairs_d = np.argwhere(cost_d >= max_dist_diff)
        if not banned_id_pairs.any():
            banned_id_pairs = banned_id_pairs_d
        else:
            banned_id_pairs = np.concatenate(
                [
                    banned_id_pairs,
                    banned_id_pairs_d,
                ]
            )

    cost = get_cost_matrix(A, B).astype(np.float32)
    if banned_id_pairs.any():
        # mark banned links as very costy
        cost[banned_id_pairs.T[0], banned_id_pairs.T[1]] = np.inf
    row_cols = np.array([])
    row_cols.resize(2, 0)
    try:
        row_cols = linear_sum_assignment_with_inf(cost)
    except ValueError:
        pass
    id_pairs = np.array(row_cols).T

    # identify unmatched predicted particles
    possible_pred_ids = np.arange(B.shape[0])
    selected_pred_ids = id_pairs.T[1]
    ghost_ids = np.setdiff1d(possible_pred_ids, selected_pred_ids).tolist()

    # identify unmatched true particles
    possible_true_ids = np.arange(A.shape[0])
    selected_true_ids = id_pairs.T[0]
    missed_ids = np.setdiff1d(possible_true_ids, selected_true_ids).tolist()

    matched_pred_ids = []
    matched_true_ids = []
    banned_id_pairs = banned_id_pairs.tolist()
    for id_pair in id_pairs.tolist():
        if id_pair in banned_id_pairs:
            missed_ids.append(id_pair[0])
            ghost_ids.append(id_pair[1])
        else:
            matched_true_ids.append(id_pair[0])
            matched_pred_ids.append(id_pair[1])

    return {
        "matched_true": np.array(matched_true_ids),
        "matched_pred": np.array(matched_pred_ids),
        "missed": np.array(missed_ids),
        "ghost": np.array(ghost_ids),
    }


def inference(
    config,
    model,
    dataset,
    xs,
    ys,
    dev_no=1,
):
    times = {}
    energies, true_energies = [], []
    positions, true_positions = [], []
    scores, classes, true_classes = [], [], []
    start_time = time.process_time()

    # INFERENCE
    config.log.debug("-> Applying predict() on testing dataset..")
    # note that this is compatible with MirroredStrategy!
    raw_preds = model.predict(dataset)
    times["Inference"] = time.process_time() - start_time

    # FIXME: TF 2.0 handles the output a bit differently...
    #        -> for some reason we do not get a tuple...
    if type(raw_preds) is not tuple:
        raw_preds = (raw_preds,)
    #        -> remove it once its understood!
    #        -> (breaking change)

    # nms
    config.log.debug("-> Applying NMS per event in the testing dataset..")
    preds = []
    for event_id, output_x in enumerate(zip(raw_preds[0])):
        boxes_xs = []
        for i in range(len(config.anchor_masks)):
            boxes_xs.append(
                config.refine_boxes(
                    np.expand_dims(output_x[i], axis=0),
                    config.anchors[config.anchor_masks[i]],
                )[:4]
            )
        preds.append(config.nms(boxes_xs))
    times["NMS"] = time.process_time() - times["Inference"] - start_time
    times["Total"] = time.process_time() - start_time

    for pred, y in zip(preds, ys):
        true_positions.append(y[..., 0:4].numpy().tolist())
        true_energies.append(y[..., 4].numpy().tolist())
        true_classes.append(y[..., 5].numpy().tolist())
        energies.append(pred[1].numpy().tolist())
        positions.append(pred[0].numpy().tolist())
        classes.append(np.argmax(pred[2].numpy(), axis=-1).tolist())
        scores.append(pred[3].numpy().tolist())

    tests = {
        # numpy does not support well ragged tensors
        "score": tf.expand_dims(tf.ragged.constant(scores, ragged_rank=1), axis=-1),
        "images": np.array(xs),
    }

    tests["pred_position"] = np.array(positions, dtype=object)
    if len(tests["pred_position"].shape) == 2:
        tests["pred_position"].resize(tests["pred_position"].shape[0], 0, 4)
        tests["pred_energy"] = np.array(energies, dtype=object)
        tests["pred_energy"].resize(tests["pred_energy"].shape[0], 0, 1)
        tests["pred_classes"] = np.array(classes, dtype=object)
        tests["pred_classes"].resize(tests["pred_energy"].shape[0], 0, 1)
    else:
        tests["pred_position"] = tf.ragged.constant(positions, ragged_rank=1)
        tests["pred_energy"] = tf.expand_dims(
            tf.ragged.constant(energies, ragged_rank=1), axis=-1
        )
        tests["pred_classes"] = tf.cast(
            tf.expand_dims(tf.ragged.constant(classes, ragged_rank=1), axis=-1),
            dtype=tf.float32,
        )

    tests["true_position"] = np.array(true_positions, dtype=object)
    if len(tests["true_position"].shape) == 2:
        tests["true_position"].resize(tests["true_position"].shape[0], 0, 4)
        tests["true_energy"] = np.array(true_energies, dtype=object)
        tests["true_energy"].resize(tests["true__energy"].shape[0], 0, 1)
        tests["true_classes"] = np.array(true_classes, dtype=object)
        tests["true_classes"].resize(tests["true_energy"].shape[0], 0, 1)
    else:
        tests["true_position"] = tf.ragged.constant(true_positions, ragged_rank=1)
        tests["true_energy"] = tf.expand_dims(
            tf.ragged.constant(true_energies, ragged_rank=1), axis=-1
        )
        tests["true_classes"] = tf.cast(
            tf.expand_dims(tf.ragged.constant(true_classes, ragged_rank=1), axis=-1),
            dtype=tf.float32,
        )

    convert_data(config, tests)

    tests["true_width"] = (
        tests["true_position"][..., 2:3] - tests["true_position"][..., 0:1]
    )
    tests["pred_width"] = (
        tests["pred_position"][..., 2:3] - tests["pred_position"][..., 0:1]
    )
    tests["true_height"] = (
        tests["true_position"][..., 3:4] - tests["true_position"][..., 1:2]
    )
    tests["pred_height"] = (
        tests["pred_position"][..., 3:4] - tests["pred_position"][..., 1:2]
    )
    tests["true_x_pos"] = (
        tests["true_position"][..., 2:3] + tests["true_position"][..., 0:1] / 2.0
    )
    tests["pred_x_pos"] = (
        tests["pred_position"][..., 2:3] + tests["pred_position"][..., 0:1] / 2.0
    )
    tests["true_y_pos"] = (
        tests["true_position"][..., 3:4] + tests["true_position"][..., 1:2] / 2.0
    )
    tests["pred_y_pos"] = (
        tests["pred_position"][..., 3:4] + tests["pred_position"][..., 1:2] / 2.0
    )

    config.log.debug("-> Looking for matched clusters..")
    truth_for_matching = [
        tests["true_x_pos"],
        tests["true_y_pos"],
        tests["true_energy"],
        # additional
        tests["true_width"],
        tests["true_height"],
        tests["true_classes"],
    ]
    if isinstance(truth_for_matching[0], tf.RaggedTensor):
        truth_for_matching = tf.concat(truth_for_matching, axis=-1).to_list()
    else:
        truth_for_matching = np.concatenate(truth_for_matching, axis=-1)

    pred_for_matching = [
        tests["pred_x_pos"],
        tests["pred_y_pos"],
        tests["pred_energy"],
        # additional
        tests["pred_width"],
        tests["pred_height"],
        tests["pred_classes"],
    ]
    if isinstance(pred_for_matching[0], tf.RaggedTensor):
        pred_for_matching = tf.concat(pred_for_matching, axis=-1).to_list()
    else:
        pred_for_matching = np.concatenate(pred_for_matching, axis=-1)

    matching = defaultdict(list)

    for a, b in zip(
        truth_for_matching,
        pred_for_matching,
    ):
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)
        if len(a.shape) == 1:
            a = np.expand_dims(a, axis=0)
        if len(b.shape) == 1:
            b = np.expand_dims(b, axis=0)
        if a.shape == (1, 0):
            a = a.reshape((0, 6))
        if b.shape == (1, 0):
            b = b.reshape((0, 6))
        ids = particle_matching(
            a[..., :3],
            b[..., :3],
            max_dist_diff=3
            * abs(config.img_x_max - config.img_x_min)
            / config.img_width,
            # TODO: when energy is ready
            # max_energy_diff=1.0,
            max_energy_diff=None,
            energy_in_log=True,
        )
        matching["matched_true"].append(a[ids["matched_true"].astype(int)])
        matching["missed"].append(a[ids["missed"].astype(int)])
        matching["matched_pred"].append(b[ids["matched_pred"].astype(int)])
        matching["ghost"].append(b[ids["ghost"].astype(int)])

    for key in matching.keys():
        matching[key] = np.concatenate(matching[key], axis=0)
    tests["matched_true_x_pos"] = matching["matched_true"][..., :1]
    tests["matched_pred_x_pos"] = matching["matched_pred"][..., :1]
    tests["matched_true_y_pos"] = matching["matched_true"][..., 1:2]
    tests["matched_pred_y_pos"] = matching["matched_pred"][..., 1:2]
    tests["matched_true_energy"] = matching["matched_true"][..., 2:3]
    tests["matched_pred_energy"] = matching["matched_pred"][..., 2:3]
    tests["matched_true_classes"] = matching["matched_true"][..., 5:]
    tests["matched_pred_classes"] = matching["matched_pred"][..., 5:]
    tests["ghost_x_pos"] = matching["ghost"][..., :1]
    tests["missed_x_pos"] = matching["missed"][..., :1]
    tests["ghost_y_pos"] = matching["ghost"][..., 1:2]
    tests["missed_y_pos"] = matching["missed"][..., 1:2]
    tests["ghost_x_energy"] = matching["ghost"][..., 2:3]
    tests["missed_x_energy"] = matching["missed"][..., 2:3]
    # for name in tests.keys():
    #     if tf.is_tensor(tests[name]):
    #         tests[name] = tests[name].numpy()

    return times, tests
