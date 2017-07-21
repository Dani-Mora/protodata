import pandas
import os
import numpy as np
import mahotas as mh

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


def hist_column(data, col_name, ax, bins, fonts=3):
    """ Plots the histogram of the given column
    Args:
        data: Pandas Dataframe
        col_name: Name of the column to plot
        ax: Matplotlib axis where to draw plot
        bins: Number of bins to use in the histograms. Will only be used
            in those columns which are numeric
        fonts: Parameter to adjust the size of the fonts in both the
            axis and the title
    """
    # Read column as data frame
    numeric = data._get_numeric_data().columns
    df = pandas.DataFrame(data[col_name])

    # Plot varies according to type
    if col_name in numeric:
        df.plot.hist(fontsize=fonts,
                     legend=False,
                     bins=bins,
                     ax=ax)
    else:
        data[col_name].value_counts().plot(fontsize=fonts,
                                           kind='bar',
                                           ax=ax)

    # Adjust title and font
    ax.set_title(col_name, fontsize=fonts)
    ax.yaxis.label.set_size(fonts)
    ax.xaxis.label.set_size(fonts)


def plot_grid(data, cols, bins=15, path=None, nrows=3, fonts=3,
              fig_size=(7, 4)):
    """ Plots the histogram of the provided columns in a grid
    Args:
        data: Pandas Dataframe
        cols: List of columns whose histograms need to be plotted
        bins: Number of bins to use in the histograms. Will only be used
            in those columns which are numeric
        path: Path where histograms will be stored as images.
            Set to None to disable
        nrows: Numer of rows to use in the grid
        fonts: Parameter to adjust the size of the fonts in both the
            axis and the title
        fig_size: Parameter to manually adjust the size of the plot
    """
    # Compute grid dimension
    ncols = int(np.ceil(len(cols) / float(nrows)))
    nrows = nrows if len(cols) > ncols else 1
    # Subplot image with margins
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
    fig.tight_layout()

    # Check correct format of the axes
    axes = axes if isinstance(axes, np.ndarray) \
        else np.array(axes)

    for i in range(0, nrows):
        for j in range(0, ncols):

            # Compute axes index and column to plot
            col_index = (i * ncols) + j
            index = np.ravel_multi_index((i, j),
                                         dims=(nrows, ncols),
                                         order='F')

            if col_index >= len(cols):
                # May be that grid is larger than number of colums
                # Remaining columns to be left blank
                fig.delaxes(axes.ravel()[index])
            else:
                # Send column to plot in the corresponding axis
                col_name = cols[col_index]
                hist_column(data,
                            col_name=col_name,
                            ax=axes.ravel()[index],
                            bins=bins,
                            fonts=fonts)

    if path is not None:
        fig.savefig(path)


def plot_image(path, ax):
    """ Shows the image in the input index in the given axis """
    im = mh.imread(path)
    ax.imshow(im)
    ax.axis('off')


def plot_image_grid(paths, path=None, nrows=3, fig_size=(7, 4)):
    """ Plots a grid containing images
    Args:
        paths: List of paths to print
        path: Path where histograms will be stored as images.
            Set to None to disable
        nrows: Numer of rows to use in the grid
        fig_size: Parameter to manually adjust the size of the plot
    """
    # Compute grid dimension
    ncols = int(np.ceil(len(paths) / float(nrows)))
    nrows = nrows if len(paths) > ncols else 1
    # Subplot image with margins
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)

    # Check correct format of the axes
    axes = axes if isinstance(axes, np.ndarray) \
        else np.array(axes)

    for i in range(0, nrows):
        for j in range(0, ncols):

            # Compute axes index and column to plot
            img_index = (i * ncols) + j
            index = np.ravel_multi_index((i, j),
                                         dims=(nrows, ncols), order='F')

            if img_index >= len(paths):
                # May be that grid is larger than number of images
                # Remaining spots to be left blank
                fig.delaxes(axes.ravel()[index])
            else:
                # Send image to plot in the corresponding axis
                plot_image(paths[img_index],
                           ax=axes.ravel()[index])

    if path is not None:
        fig.savefig(path)


def plot_histograms(data, var, path, prefix, nrows=3, grid=8, bins=25,
                    fonts=3, fig_size=(7, 4)):
    """ Plots the given columns as histograms into a set of grids of
    plots and can stores the data into a certain location

    Args:
        data: Pandas Dataframe
        var: List of variables whose histograms need to be plotted
        path: Path where histograms will be stored as images.
            Set to None to disable
        prefix: Prefix to add to the image paths. Will be skipped if path
            is None
        nrows: Numer of rows to use in the grid
        grid: Size of the grid. Several grids will be displayed until all
            plots are shown.
        bins: Number of bins to use in the histograms. Will only be used
            in those columns which are numeric
        fonts: Parameter to adjust the size of the fonts in both the axis
            and the title
        fig_size: Parameter to manually adjust the size of the plot
    """
    for i in range(0, len(var), grid):
        # Plot using sets of columns
        if i + grid > len(var):
            point = len(var)
        else:
            point = i + grid

        # Set storing path, if requested
        if path is None:
            fig_path = None
        else:
            fig_path = os.path.join(path,
                                    '_'.join([prefix, str(i), str(point)])
                                    + '.png')
        # Send to display
        plot_grid(data,
                  cols=var[i:point],
                  path=fig_path,
                  bins=bins,
                  fonts=fonts,
                  nrows=nrows,
                  fig_size=fig_size)


def unify_columns(data, name1, name2, new_name, col_index):
    """ Unify two columns using boolean disjunction
    Args:
        data: Pandas DataFrame
        name1: Name of the first column
        name2: Name of the second column
        new_name: Name of the new column to introduce
        col_index: Column index where new column will be set
    """

    def combiner(x, y):
        return x or y

    first_version = data[name1]
    second_version = data[name2]
    # Combine both
    new_version = first_version.combine(second_version, combiner)
    data.insert(col_index, new_name, new_version)
    # Delete the others
    data.drop(name1, axis=1, inplace=True)
    data.drop(name2, axis=1, inplace=True)
