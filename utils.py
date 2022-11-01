import logging
import sys


class CustomFormatter(logging.Formatter):

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, *args, prefix_format="", **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_format = prefix_format

    def format(self, record):
        log_fmt = f"{self.FORMATS.get(record.levelno)}{self.prefix_format}{self.RESET}"
        if record.levelno >= logging.WARNING:
            log_fmt += " - (%(filename)s:%(lineno)d)"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def activate_logger(
    directory: str = None,
    logger_level: str = "INFO",
    root_logger_level: str = "CRITICAL",
    stdout=None,
):
    chosen_stdout = stdout
    if not chosen_stdout:
        chosen_stdout = sys.stdout
    handlers = [
        logging.StreamHandler(chosen_stdout),
    ]
    prefix_format = "%(levelname)s: %(message)s"
    if directory:
        logger_file = f"{directory}/output.log"
        print("Activating the logger in " + logger_file)
        handlers.append(logging.FileHandler(logger_file))
        prefix_format = "%(asctime)s - %(levelname)s: %(message)s"

    handlers[0].setFormatter(CustomFormatter(prefix_format=prefix_format))
    logging.basicConfig(
        handlers=handlers,
        format=prefix_format,  # for default
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=getattr(logging, root_logger_level),
    )
    logger_level = getattr(logging, logger_level)
    logger = logging.getLogger("MCRecoCalo")
    logger.setLevel(logger_level)
    return logger
