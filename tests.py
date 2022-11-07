import numpy as np
import time
from collections import defaultdict
import tensorflow as tf
from scipy.optimize import linear_sum_assignment


def ragged_to_normal(np_array):
    return np.array(
        [elem for elem in np_array if not (type(elem) == np.ndarray and elem.size == 0)]
    )


def convert_data(config, tests):
    tests["pred_energy"] = config.convert_to_energy(tests["pred_energy"])
    tests["true_energy"] = config.convert_to_energy(tests["true_energy"])
    for position_type in ["pred_position", "true_position"]:
        tests[position_type][..., 0] = config.convert_to_position(
            tests[position_type][..., 0], dim="x"
        )
        tests[position_type][..., 2] = config.convert_to_position(
            tests[position_type][..., 2], dim="x"
        )
        tests[position_type][..., 1] = config.convert_to_position(
            tests[position_type][..., 1], dim="y"
        )
        tests[position_type][..., 3] = config.convert_to_position(
            tests[position_type][..., 3], dim="y"
        )
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
        A_e = A[..., 2:3]
        B_e = B[..., 2:3]
        if energy_in_log:
            if (A_e <= 0.0).any() or (B_e <= 0.0).any():
                raise ValueError("Energy must be >= 0.")
            A_e = np.log10(A_e)
            B_e = np.log10(B_e)
        cost_e = np.sqrt(get_cost_matrix(A_e, B_e, ndim=1))
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
        A_d = A[..., 0:2]
        B_d = B[..., 0:2]
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
    row_cols = linear_sum_assignment_with_inf(cost)
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


def prepare_dataset_for_inference(
    config,
    image_transformation,
    model,
    dataset,
    samples=100,
):

    times = {}
    energies, true_energies = [], []
    positions, true_positions = [], []
    scores, classes, true_classes = [], [], []

    start_time = time.process_time()
    # dataset
    config.log.debug("-> Fetching testing dataset..")
    xs, ys = [], []
    for x, y in dataset.take(samples) if samples > 0 else dataset:
        xs.append(np.expand_dims(x, 0))
        ys.append(y)
    times["Dataset Reading"] = time.process_time() - start_time

    # preprocessing
    config.log.debug("-> Preprocessing testing dataset..")
    merged_xs = np.concatenate(xs, 0)
    merged_xs = image_transformation(merged_xs)
    times["Preprocessing"] = time.process_time() - times["Dataset Reading"] - start_time

    # inference on the whole dataset
    config.log.debug("-> Applying predict() on testing dataset..")
    model.apply_refine = True
    raw_preds = model.predict(merged_xs, steps=len(xs))
    times["Inference"] = time.process_time() - times["Preprocessing"] - start_time
    model.apply_refine = False

    # nms
    config.log.debug("-> Applying NMS per event in the testing dataset..")
    preds = []
    for event in range(len(xs)):
        pre_nms = [
            [pred[0][event], pred[1][event], pred[2][event], pred[3][event]]
            for pred in raw_preds
        ]
        preds.append(model.nms(pre_nms))
    times["NMS"] = time.process_time() - times["Inference"] - start_time
    times["Total"] = time.process_time() - start_time

    for pred, y in zip(preds, ys):
        true_positions.append(y[..., 0:4].numpy())
        true_energies.append(y[..., 4].numpy())
        true_classes.append(y[..., 5].numpy())
        energies.append(pred[1].numpy())
        positions.append(pred[0].numpy())
        classes.append(np.argmax(pred[2].numpy(), axis=-1))
        scores.append(pred[3].numpy())

    tests = {
        "pred_energy": np.array(energies, dtype=object),
        "true_energy": np.array(true_energies, dtype=object),
        "score": np.array(scores, dtype=object),
        "pred_position": np.array(positions, dtype=object),
        "true_position": np.array(true_positions, dtype=object),
        "true_classes": np.array(true_classes, dtype=object),
        "pred_classes": np.array(classes, dtype=object),
        "images": np.array(xs),
    }

    if len(tests["pred_energy"].shape) == 1:
        tests["pred_energy"].resize(tests["pred_energy"].shape[0], 0)
    if len(tests["pred_position"].shape) == 1:
        tests["pred_position"].resize(tests["pred_position"].shape[0], 0, 4)
    if len(tests["pred_classes"].shape) == 1:
        tests["pred_classes"].resize(tests["pred_classes"].shape[0], 0)

    convert_data(config, tests)

    tests["true_width"] = (
        tests["true_position"][..., 2] - tests["true_position"][..., 0]
    )
    tests["pred_width"] = (
        tests["pred_position"][..., 2] - tests["pred_position"][..., 0]
    )
    tests["true_height"] = (
        tests["true_position"][..., 3] - tests["true_position"][..., 1]
    )
    tests["pred_height"] = (
        tests["pred_position"][..., 3] - tests["pred_position"][..., 1]
    )
    tests["true_x_pos"] = (
        tests["true_position"][..., 2] + tests["true_position"][..., 0] / 2.0
    )
    tests["pred_x_pos"] = (
        tests["pred_position"][..., 2] + tests["pred_position"][..., 0] / 2.0
    )
    tests["true_y_pos"] = (
        tests["true_position"][..., 3] + tests["true_position"][..., 1] / 2.0
    )
    tests["pred_y_pos"] = (
        tests["pred_position"][..., 3] + tests["pred_position"][..., 1] / 2.0
    )

    config.log.debug("-> Looking for matched clusters..")
    truth_for_matching = np.concatenate(
        [
            tests["true_x_pos"],
            tests["true_y_pos"],
            tests["true_energy"],
            # additional
            tests["true_width"],
            tests["true_height"],
            tests["true_classes"],
        ],
        axis=-1,
    )
    pred_for_matching = np.concatenate(
        [
            tests["pred_x_pos"],
            tests["pred_y_pos"],
            tests["pred_energy"],
            # additional
            tests["pred_width"],
            tests["pred_height"],
            tests["pred_classes"],
        ],
        axis=-1,
    )
    matching = defaultdict(list)
    for a, b in zip(truth_for_matching, pred_for_matching):
        if len(a.shape) == 1:
            a = np.expand_dims(a, axis=0)
        if len(b.shape) == 1:
            b = np.expand_dims(b, axis=0)
        if a.shape == (1, 0):
            a = a.reshape((0, 3))
        if b.shape == (1, 0):
            b = b.reshape((0, 3))
        ids = particle_matching(
            a[..., :3],
            b[..., :3],
            # TODO: important!
            # max_dist_diff =
            # max_energy_diff =
        )
        matching["matched_true"].append(a[ids["matched_true"].astype(int)])
        matching["missed"].append(a[ids["missed"].astype(int)])
        matching["matched_pred"].append(b[ids["matched_pred"].astype(int)])
        matching["ghost"].append(b[ids["ghost"].astype(int)])

    for key in matching.keys():
        matching[key] = np.concatenate(matching[key])
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

    return times, tests
