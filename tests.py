import numpy as np
import time
import tensorflow as tf

def ragged_to_normal(np_array):
    return np.array([elem for elem in np_array if not (type(elem) == np.ndarray and elem.size == 0)])

def convert_data(config, tests):
    tests["pred_energy"] = config.convert_to_energy(tests["pred_energy"])
    tests["true_energy"] = config.convert_to_energy(tests["true_energy"])
    tests["pred_position"] = config.convert_to_position(tests["pred_position"])
    tests["true_position"] = config.convert_to_position(tests["true_position"])


def prepare_dataset_for_inference(
    log,
    image_transformation,
    model,
    dataset,
    samples=100,
):

    times = {}
    energies, true_energies = [], []
    positions, true_positions = [], []
    scores = []

    start_time = time.process_time()
    # dataset
    log.debug("-> Fetching testing dataset..")
    xs, ys = [], []
    for x, y in dataset.take(samples) if samples > 0 else dataset:
        xs.append(np.expand_dims(x, 0))
        ys.append(y)
    times["Dataset Reading"] = time.process_time() - start_time

    # preprocessing
    log.debug("-> Preprocessing testing dataset..")
    merged_xs = np.concatenate(xs, 0)
    merged_xs = image_transformation(merged_xs)
    times["Preprocessing"] = time.process_time() - times["Dataset Reading"] - start_time

    # inference on the whole dataset
    log.debug("-> Applying predict() on testing dataset..")
    model.apply_refine = True
    raw_preds = model.predict(merged_xs, steps=len(xs))
    times["Inference"] = time.process_time() - times["Preprocessing"] - start_time
    model.apply_refine = False

    # nms
    log.debug("-> Applying NMS per event in the testing dataset..")
    preds = []
    for event in range(len(xs)):
        pre_nms = [[pred[0][event],
                    pred[1][event],
                    pred[2][event],
                    pred[3][event]] for pred in raw_preds]
        preds.append(model.nms(pre_nms))
    times["NMS"] = time.process_time() - times["Inference"] - start_time
    times["Total"] = time.process_time() - start_time

    longest_txt = max([len(key) for key in times.keys()])
    for txt, seconds in times.items():
        msg = f"--> {txt} time:{' ' * (longest_txt - len(txt))} {round(seconds, 3)} s,"
        msg += f" {round(seconds * 1000 / len(xs), 3)} ms/event"
        msg += f" {round(seconds / times['Total'] * 100, 3)} % total time"
        log.debug(msg)


    for pred, y in zip(preds, ys):
        true_positions.append(y[..., 0:4].numpy())
        true_energies.append(y[..., 5].numpy())
        energies.append(pred[1].numpy())
        positions.append(pred[0].numpy())
        scores.append(pred[3].numpy())

    return {
        "pred_energy": np.array(energies, dtype=object),
        "true_energy": np.array(true_energies, dtype=object),
        "score": np.array(scores, dtype=object),
        "pred_position": np.array(positions, dtype=object),
        "true_position": np.array(true_positions, dtype=object),
    }
