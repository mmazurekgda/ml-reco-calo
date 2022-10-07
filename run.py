import argparse
import logging as log
import pyfiglet as pf
import sys
import os
from datetime import datetime

# local
from cnn.config import Config as CNNConfig
from cnn.run import CNN
from utils import activate_logger


def parse_options(args, parsed_start_time: str) -> CNNConfig:
    # create the working directory
    directory = args.directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    # activate the logger
    activate_logger(
        directory=directory,
        parsed_start_time=parsed_start_time,
        logger_level=args.verbosity
    )
    log.info(f"\n{header}")
    log.info("Start time: " + parsed_start_time)
    log.debug("Parsing options MCRecoCalo")
    config = CNNConfig()
    for prop, def_value in CNNConfig.OPTIONS.items():
        overriden_text = ""
        value = def_value
        if prop in args:
            parsed_value = getattr(args, prop)
            if parsed_value:
                overriden_text = "(OVERRIDEN)"
                value = parsed_value
        log.debug(f"{prop}: {value} {overriden_text}")
        setattr(config, prop, value)
    return config



if __name__ == "__main__":
    start_time = datetime.utcnow()
    parsed_start_time = start_time.strftime("%Y%m%d_%H%M%S%f")
    header = pf.figlet_format("MCRecoCalo")
    parser = argparse.ArgumentParser(
        description=header,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--training',
        help='training on',
        action='store_false',
    )
    parser.add_argument(
        '--inference',
        help='inference on',
        action='store_false',
    )
    parser.add_argument(
        '--directory',
        help='directory where to keep files',
        default=f"./{parsed_start_time}",
    )
    parser.add_argument(
        '--type',
        help='type of framework',
        default='CNN',
        choices=['CNN', 'GNN'],
    )
    parser.add_argument(
        '--verbosity',
        help='level of verbosity',
        default='INFO',
        choices=['INFO', 'DEBUG'],
    )

    for name, value in CNNConfig.OPTIONS.items():
        prop_type = type(value)
        if prop_type in [int, str, bool, list, float]:
            parser.add_argument(f'--{name}', type=prop_type)

    args = parser.parse_args()
    config = parse_options(args, parsed_start_time)

    framework = CNN()
    # if args.training:
    #     framework.train()




