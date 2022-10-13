import pandas as pd
import logging as log
import pyfiglet as pf
import argparse
import sys
import os

from root_to_pandas import convert_root_to_pandas

def convert_root_to_hdf5(
    root_filename,
    complevel=None, # {0-9}, default None
    complib='zlib', # {‘zlib’, ‘lzo’, ‘bzip2’, ‘blosc’}, default ‘zlib’
):
    pandas_dfs = convert_root_to_pandas(root_filename)
    work_dir = os.path.dirname(os.path.abspath(root_filename))
    logger = log.getLogger('MCRecoCalo')
    for key, pandas_df in pandas_dfs.items():
        hdf5_filename = key.replace('/', '_')
        hdf5_filename =  hdf5_filename.split(';')[0]
        hdf5_filepath = "{}/{}.h5".format(work_dir, hdf5_filename)
        logger.debug("Saving to " + hdf5_filepath)
        pandas_df.to_hdf(
            hdf5_filepath,
            'df',
            complevel=complevel,
            complib=complib,
        )


if __name__ == "__main__":
    header = pf.figlet_format("ROOT TO HDF5")
    parser = argparse.ArgumentParser(
        description=header,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--rootfilepath',
        help='Path to the root file',
        required=True,
    )
    parser.add_argument(
        '--compressionlevel',
        help='{0-9}',
        type=int,
    )
    parser.add_argument(
        '--compressionlib',
        help='‘zlib’, ‘lzo’, ‘bzip2’, ‘blosc’',
        default='zlib',
        type=str,
    )
    parser.add_argument(
        '--verbosity',
        help='level of verbosity',
        default='INFO',
        choices=['INFO', 'DEBUG'],
    )

    args = parser.parse_args()

    log.basicConfig(
        handlers=[
            log.StreamHandler(sys.stdout),
            log.FileHandler("./root_to_hdf5.log"),
        ],
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=getattr(log, args.verbosity),
    )

    logger = log.getLogger('MCRecoCalo')

    logger.info(f"\n{header}")

    convert_root_to_hdf5(
        args.rootfilepath,
        complevel=args.compressionlevel,
        complib=args.compressionlib,
    )

    logger.info("Done")
