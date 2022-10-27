from matplotlib import pyplot as plt
from cnn.utils import x_cell_to_pos, y_cell_to_pos


def plot_shapes(ax, df, cluster_type="", **kwargs):
    # FIXME: don't like this zipping of colums, but it seems to be the fastest for now
    # cont = df.copy()
    # cont['Cell_X_Min'] = x_cell_to_pos(cont['Cell_X_Min_Cluster' + cluster_type])
    # cont['Cell_X_Max'] = x_cell_to_pos(cont['Cell_X_Max_Cluster' + cluster_type])
    # cont['Cell_Y_Min'] = y_cell_to_pos(cont['Cell_Y_Min_Cluster' + cluster_type])
    # cont['Cell_Y_Max'] = y_cell_to_pos(cont['Cell_Y_Max_Cluster' + cluster_type])

    for min_x, max_x, min_y, max_y in zip(
        df["Cell_X_Min" + cluster_type],
        df["Cell_X_Max" + cluster_type],
        df["Cell_Y_Min" + cluster_type],
        df["Cell_Y_Max" + cluster_type],
    ):
        ax.add_artist(
            plt.Rectangle(
                (min_x, min_y),
                height=(max_y - min_y),
                width=(max_x - min_x),
                fill=False,
                **kwargs
            )
        )
