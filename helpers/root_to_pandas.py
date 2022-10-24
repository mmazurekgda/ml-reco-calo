import uproot
import logging as log
import os


def convert_root_to_pandas(
    root_filename,
) -> dict:
    root_file = uproot.open(root_filename)
    logger = log.getLogger("MCRecoCalo")
    logger.debug("Opened: " + root_filename)
    work_dir = os.path.dirname(os.path.abspath(root_filename))
    pandas_dfs = {}
    for key, value in root_file.items():
        logger.debug("Found key: " + key)
        if "/" in key:  # TODO: not the best way, but whatever...
            logger.debug("Is a TTree.")
            df = value.arrays(value.keys(), library="pd")
            logger.debug("Converted to pd/dataframe")
            logger.debug(df.head())
            pandas_dfs[key] = df
    return pandas_dfs
