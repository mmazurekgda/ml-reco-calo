import numpy as np
import pandas as pd
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
           log_fmt +=  " - (%(filename)s:%(lineno)d)"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def activate_logger(
    directory: str = None,
    logger_level: str = 'INFO',
    root_logger_level: str = 'WARN',
    stdout = None,
):
    chosen_stdout = stdout
    if not chosen_stdout:
        chosen_stdout = sys.stdout
    handlers = [
        logging.StreamHandler(chosen_stdout),
    ]
    prefix_format="%(levelname)s: %(message)s"
    if directory:
        logger_file = f"{directory}/output.log"
        print("Activating the logger in " + logger_file)
        handlers.append(logging.FileHandler(logger_file))
        prefix_format="%(asctime)s - %(levelname)s: %(message)s"

    handlers[0].setFormatter(CustomFormatter(prefix_format=prefix_format))
    logging.basicConfig(
        handlers=handlers,
        format=prefix_format, # for default
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=getattr(logging, root_logger_level),
    )
    logger_level = getattr(logging, logger_level)
    logger = logging.getLogger('MCRecoCalo')
    logger.setLevel(logger_level)
    return logger

def x_cell_to_pos(x, grid_x_cells, grid_x_min, grid_x_max):
    return x / grid_x_cells * (grid_x_max - grid_x_min) + grid_x_min

def y_cell_to_pos(y, grid_y_cells, grid_y_min, grid_y_max):
    return y / grid_y_cells * (grid_y_max - grid_y_min) + grid_y_min

def clusterize_barycenter(
    particles_df,
    hits_df,
    new_suffix=None,
    min_cell_size_no=3,
    max_cell_size_no=385,
    min_energy_ratio=1.
):
    hits['Cell_X_Central'] = hits['Cell_X'] * hits['Active_Energy'] / hits['Active_Energy_Sum']
    hits['Cell_Y_Central'] = hits['Cell_Y'] * hits['Active_Energy'] / hits['Active_Energy_Sum']
    hits_for_barycenter = hits[[
        'Event_ID',
        'Particle_Index',
        'Cell_X_Central',
        'Cell_Y_Central',
    ]].groupby(['Event_ID', 'Particle_Index'], as_index=False)[[
        'Cell_X_Central',
        'Cell_Y_Central',
    ]].sum().copy()
    hits.drop(columns=[
        'Cell_X_Central',
        'Cell_Y_Central',
    ], inplace=True)
    hits = hits.merge(
        hits_for_barycenter[[
            'Event_ID',
            'Particle_Index',
            'Cell_X_Central',
            'Cell_Y_Central'
        ]],
        left_on=['Event_ID', 'Particle_Index'],
        right_on=['Event_ID', 'Particle_Index'],
        suffixes=(None, None)
    )

def weighted_mean(
    values_col="Values",
    weights_col='Weights',
    new_column="Mean",
):
    def _call(input_df):
        weights = input_df[weights_col]
        vals = input_df[values_col]
        wavg = np.average(vals, weights=weights)
        return pd.Series({
            new_column: wavg,
        })
    return _call

def weighted_std(
    values_col="Values",
    weights_col='Weights',
    new_column="Stddev",
):
    def _call(input_df):
        weights = input_df[weights_col]
        vals = input_df[values_col]
        wavg = np.average(vals, weights=weights)
        numer = np.sum(weights * (vals - wavg) ** 2)
        count = vals.count()
        denom = ((count - 1) / count) * np.sum(weights)
        return pd.Series({
            new_column: np.sqrt(numer / denom),
        })
    return _call

def clusterize(
    particles_df,
    hits_df,
    compute_barycenter=False, # compute barycenter no matter what
    new_suffix=None,
    # extra cuts
    min_cell_size_no=0,
    max_cell_size_no=np.inf,
    min_energy_ratio=1.,
    # barycenter options
    use_barycenter=False,     # min/max cell or +/- stddev
    stddev_no=1,              # how many stddevs
    min_barycenter_cell_size_no=1,
    img_x_min = -np.inf,
    img_x_max =  np.inf,
    img_y_min = -np.inf,
    img_y_max =  np.inf,
):
    # simulate some kind of cellular automaton on MC Truth
    # using DataFrame indexing & sorting it takes now ~50ms / event
    #
    #
    # find the max and sum of all the hits in each cluster
    cols_to_groupby = [
        "Event_ID",
        "Particle_Index",
    ]
    cols_to_select = cols_to_groupby + ["Active_Energy"]
    hits_grouped = hits_df[cols_to_select].groupby(
        by=cols_to_groupby,
        as_index=False
    )
    energy_max = hits_grouped.max()
    energy_sum = hits_grouped.sum()
    # find a central cell, i.e. one with the largest energy deposit
    central_cells = hits_df.merge(energy_max)
    hits = hits_df.merge(
        central_cells,
        on=cols_to_groupby,
        suffixes=('', '_Central')
    )
    hits = hits.merge(
        energy_sum,
        on=cols_to_groupby,
        suffixes=('', '_Sum')
    )

    # let's reject the hits that do not meet the conditions
    # compute a taxi-cab distance between other cells and the central cell
    hits["Distance"] = abs(hits["Cell_X"] - hits["Cell_X_Central"]) + abs(hits["Cell_Y"] - hits["Cell_Y_Central"])
    hits["IsMinCluster"] = (hits["Distance"] < min_cell_size_no * hits["Cell_Size_Central"])
    hits["IsMaxCluster"] =  (hits["Distance"] < max_cell_size_no * hits["Cell_Size_Central"])
    # sort the hits, distance ascending, energy descending
    hits = hits.sort_values(['Event_ID', 'Particle_Index', "Distance", "Active_Energy"], ascending=[1, 1, 1, 0])
    hits["Energy_Cum"] = hits[['Event_ID', 'Particle_Index', 'Active_Energy']].groupby(['Event_ID', 'Particle_Index']).cumsum()
    hits["IsCumEnergyLessMinEnergyRatio"] = hits["Energy_Cum"] / hits["Active_Energy_Sum"] < min_energy_ratio
    # create a cluster by selecting those hits that are within 3 (or other) cell size radius or conrtibute to 0.9 (or other) of total energy deposit
    hits = hits[(hits["IsCumEnergyLessMinEnergyRatio"] | hits["IsMinCluster"]) & hits["IsMaxCluster"]]
    hits.drop(columns=['Active_Energy_Sum'], inplace=True)

    hits_energy = hits[cols_to_select].groupby(
        by=cols_to_groupby,
        as_index=False
    ).sum()

    hits_energy.columns = [col + (new_suffix if col not in cols_to_groupby else "")  for col in hits_energy.columns]

    particles_df = particles_df.merge(
        hits_energy,
        on=cols_to_groupby,
    )

    hits_central = hits[cols_to_groupby + ['Cell_X_Central', 'Cell_Y_Central', 'Cell_Size_Central']].groupby(
        by=cols_to_groupby,
        as_index=False
    ).max()

    hits_central.columns = [col + (new_suffix if col not in cols_to_groupby else "")  for col in hits_central.columns]

    particles_df = particles_df.merge(
        hits_central,
        on=cols_to_groupby,
    )

    # barycenter again on the selected hits
    if compute_barycenter or use_barycenter:
        hits['Position_X'] = hits['Cell_X'] + hits['Cell_Size'] / 2.
        hits['Position_Y'] = hits['Cell_Y'] + hits['Cell_Size'] / 2.

        cols_to_select_std = cols_to_select + [
            'Position_X',
            'Position_Y',
        ]

        hits_grouped_std = hits[cols_to_select_std].groupby(
            by=cols_to_groupby,
            as_index=False,
        )

        hits_barycenter_stdx = hits_grouped_std.apply(weighted_std(
            values_col="Position_X",
            weights_col='Active_Energy',
            new_column="Barycenter_X_stddev",
        ))

        hits_barycenter_meanx = hits_grouped_std.apply(weighted_mean(
            values_col="Position_X",
            weights_col='Active_Energy',
            new_column="Barycenter_X",
        ))

        hits_barycenter = hits_barycenter_stdx.merge(
            hits_barycenter_meanx,
            on=cols_to_groupby,
        )

        hits_barycenter_stdy = hits_grouped_std.apply(weighted_std(
            values_col="Position_Y",
            weights_col='Active_Energy',
            new_column="Barycenter_Y_stddev",
        ))


        hits_barycenter = hits_barycenter.merge(
            hits_barycenter_stdy,
            on=cols_to_groupby,
        )

        hits_barycenter_meany = hits_grouped_std.apply(weighted_mean(
            values_col="Position_Y",
            weights_col='Active_Energy',
            new_column="Barycenter_Y",
        ))

        hits_barycenter = hits_barycenter.merge(
            hits_barycenter_meany,
            on=cols_to_groupby,
        )

        hits_barycenter.columns = [col + (new_suffix if col not in cols_to_groupby else "")  for col in hits_barycenter.columns]

        particles_df = particles_df.merge(
            hits_barycenter,
            on=cols_to_groupby,
        )


    if use_barycenter:
        # cluster spanned by the +/- weighted stddev
        std_from_cell = min_barycenter_cell_size_no * particles_df['Cell_Size_Central' + new_suffix] / 2.

        particles_df['xstddev'] = stddev_no * particles_df['Barycenter_X_stddev' + new_suffix]
        x_smaller_or_null = (particles_df['xstddev'].isna() | (particles_df['xstddev'] < std_from_cell))
        x_smaller_or_null = (x_smaller_or_null | particles_df['xstddev'].isin([np.nan, np.inf, -np.inf]))
        particles_df.loc[x_smaller_or_null, 'xstddev'] = std_from_cell

        particles_df['ystddev'] = stddev_no * particles_df['Barycenter_Y_stddev' + new_suffix]
        y_smaller_or_null = (particles_df['ystddev'].isna() | (particles_df['ystddev'] < std_from_cell))
        y_smaller_or_null = (y_smaller_or_null | particles_df['ystddev'].isin([np.nan, np.inf, -np.inf]))
        particles_df.loc[y_smaller_or_null, 'ystddev'] = std_from_cell

        particles_df['Cell_X_Min' + new_suffix] = particles_df['Barycenter_X' + new_suffix] - particles_df['xstddev']
        particles_df['Cell_X_Max' + new_suffix] = particles_df['Barycenter_X' + new_suffix] + particles_df['xstddev']
        particles_df['Cell_Y_Min' + new_suffix] = particles_df['Barycenter_Y' + new_suffix] - particles_df['ystddev']
        particles_df['Cell_Y_Max' + new_suffix] = particles_df['Barycenter_Y' + new_suffix] + particles_df['ystddev']


        particles_df.loc[particles_df['Cell_X_Min' + new_suffix] < img_x_min, 'Cell_X_Min' + new_suffix] = img_x_min
        particles_df.loc[particles_df['Cell_X_Max' + new_suffix] > img_x_max, 'Cell_X_Max' + new_suffix] = img_x_max
        particles_df.loc[particles_df['Cell_Y_Min' + new_suffix] < img_y_min, 'Cell_Y_Min' + new_suffix] = img_y_min
        particles_df.loc[particles_df['Cell_Y_Max' + new_suffix] > img_y_max, 'Cell_Y_Max' + new_suffix] = img_y_max

        particles_df.drop(columns=['xstddev', 'ystddev'], inplace=True)

    else:
        # cluster spanned by the cells on the boundary
        hits['Cell_X_Extended'] = hits['Cell_X'] + hits['Cell_Size']
        hits['Cell_Y_Extended'] = hits['Cell_Y'] + hits['Cell_Size']
        hits = hits.groupby(['Event_ID', 'Particle_Index'])[['Cell_X', 'Cell_Y', 'Cell_X_Extended', 'Cell_Y_Extended']].agg(['max', 'min'])
        hits = hits.drop(columns=[
            ('Cell_X', 'max'),
            ('Cell_Y', 'max'),
            ('Cell_X_Extended', 'min'),
            ('Cell_Y_Extended', 'min')
        ])
        hits.columns = [('_'.join([col.title() for col in cols])).strip().replace("_Extended", "") for cols in hits.columns.values]
        hits.columns = [col + (new_suffix if col not in cols_to_groupby else "")  for col in hits.columns]
        particles_df = particles_df.merge(
            hits,
            on=cols_to_groupby,
        )

    return particles_df
