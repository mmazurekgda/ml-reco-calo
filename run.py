import argparse
import logging
import pyfiglet as pf
import sys
import os
from datetime import datetime
from importlib import import_module

# local
from utils import activate_logger
from cnn.config import Config as CNNConfig


def parse_options(args, parsed_start_time: str) -> CNNConfig:
    # create the working directory
    directory = args.output_area
    if not os.path.exists(directory):
        os.makedirs(directory)
    # activate the logger
    log = activate_logger(
        directory=directory,
        logger_level=args.verbosity,
    )
    log.info(f"\n{header}")
    log.info("Start time: " + parsed_start_time)
    log.debug("Parsing options MCRecoCalo")
    config = CNNConfig(
        output_area=directory,
        load_config_file=args.config_file,
    )
    for prop, def_value in CNNConfig.OPTIONS.items():
        overriden_text = ""
        value = def_value
        if prop in args:
            parsed_value = getattr(args, prop)
            if parsed_value:
                overriden_text = "(MANUALLY OVERRIDEN)"
                value = parsed_value
                # FIXME: this looks tedious
                if type(value) is str:
                    if value.strip()[0] == "[":
                        value = list(eval(value))
                    if value in ["False", "True"]:
                        value = bool(eval(value))
        if overriden_text or not args.config_file:
            log.debug(f"-> {prop}: {value} {overriden_text}")
            setattr(config, prop, value)
    return config


def model_lookup(model_name):
    log = logging.getLogger("MCRecoCalo")
    if not model_name:
        raise TypeError("Model must be provided!")
    model = None
    models_location = "cnn.models"
    this_model_location = ".".join([models_location, model_name])
    try:
        model = getattr(import_module(this_model_location), "Model")
        log.debug(f"-> Imported '{model_name}' from '{models_location}'.")
    except AttributeError:
        log.error("-> No model() definition in your model!")
    except ModuleNotFoundError:
        log.error(f"-> Could not find '{model_name}' in '{models_location}'")
    return model


def dataloader_lookup(dataloader_name):
    log = logging.getLogger("MCRecoCalo")
    dataloader = None
    if not dataloader_name:
        log.debug("-> No dataloader specified. Using the default TFRecords loader.")
        dataloader_name = "tfrecords_loader"
    loaders_location = "dataloaders"
    this_loader_location = ".".join([loaders_location, dataloader_name])
    try:
        dataloader = getattr(import_module(this_loader_location), "DataLoader")
        log.debug(f"-> Imported '{dataloader_name}' from '{loaders_location}'.")
    except AttributeError:
        log.error("-> No dataloader() definition in your file!")
    except ModuleNotFoundError:
        log.error(f"-> Could not find '{dataloader_name}' in '{loaders_location}'")
    return dataloader


if __name__ == "__main__":
    start_time = datetime.utcnow()
    parsed_start_time = start_time.strftime("%Y%m%d_%H%M%S%f")
    header = pf.figlet_format("MCRecoCalo")
    parser = argparse.ArgumentParser(
        description=header,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # main options for running the program from cmd

    parser.add_argument(
        "--model_name",
        help="model name (file name)",
        required=True,
    )

    parser.add_argument(
        "--dataloader_name",
        help="dataloader name (file name)",
    )

    parser.add_argument(
        "--training",
        help="training mode",
        action="store_true",
    )
    # TODO: not yet
    # parser.add_argument(
    #     '--inference',
    #     help='inference on',
    #     action='store_false',
    # )

    parser.add_argument(
        "--config_file",
        help="preload the configuration file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_area",
        help="directory where to keep files",
        default=f"./evaluations/{parsed_start_time}",
    )
    # TODO: not yet
    # parser.add_argument(
    #     '--type',
    #     help='type of framework',
    #     default='CNN',
    #     choices=['CNN', 'GNN'],
    # )
    parser.add_argument(
        "--verbosity",
        help="level of verbosity",
        default="INFO",
        choices=["INFO", "DEBUG"],
    )
    parser.add_argument(
        "--mirrored_strategy",
        help="training in mirrored strategy",
        action="store_true",
    )

    # optional, provide support for config options
    for name, value in CNNConfig.CONFIGURABLE_OPTIONS.items():
        prop_type = type(value)
        if prop_type in [int, str, float]:
            parser.add_argument(f"--{name}", type=prop_type)
        elif prop_type is bool:
            parser.add_argument(f"--{name}", type=str, choices=["True", "False"])
        elif prop_type is list:
            parser.add_argument(f"--{name}", type=str)
            # parser.add_argument(f'--{name}', action='store_true')
            # parser.add_argument(f'--no-{name}', dest=name, action='store_false')
            # parser.set_defaults(feature=True)

    args = parser.parse_args()
    config = parse_options(args, parsed_start_time)
    log = logging.getLogger("MCRecoCalo")

    log.info("Loading model...")
    model = model_lookup(args.model_name)

    log.info("Loading dataloader...")
    dataloader = dataloader_lookup(args.dataloader_name)

    log.info("Moving to the main part.")
    config._freeze()

    log.info("Instantiating the framework...")
    from cnn.run import CNN
    import tensorflow as tf  # FIXME: import here, but already added before

    dev_no = 1
    if args.mirrored_strategy:
        dev_no = len(tf.config.experimental.list_physical_devices("GPU"))

    framework = CNN(
        dataloader=dataloader,
        model=model,
        config=config,
        dev_no=dev_no,
    )
    # check some conflicting options
    config.check_compatibility()

    log.debug("Dumping the config file...")
    config.dump_to_file()

    if args.training:
        log.info("The chosen main action is: TRAINING")
        if args.mirrored_strategy:
            with tf.distribute.MirroredStrategy().scope():
                framework.train()
        else:
            framework.train()
    else:
        log.error("No main action selected.")
    end_time = datetime.utcnow()
    execution_time = (end_time - start_time).total_seconds()
    log.info("End of the program.")
    log.info(f"Finished at {end_time.strftime('%Y%m%d_%H%M%S%f')}.")
    log.info(f"Took {execution_time} s.")
