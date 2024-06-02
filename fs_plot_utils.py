import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab
import seaborn as sns

def set_axis_infos(
    ax,
    xlabel: str = None,
    ylabel: int = None,
    xlim=None,
    ylim=None,
    legend=None,
    title_str: str = None,
    xticks=None,
    yticks=None,
    xlabel_size: int = 20,
    ylabel_size: int = 20,
    title_size: int = 26,
    ticks_size: int = 18,
    legend_size: int = 20,
    legend_loc: str = "best",
    ) -> None:
    """Set axis information.

    :param ax: axis
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limits
    :param ylim: y-axis limits
    :param legend: legend
    :param title_str: title
    :param xticks: x-axis ticks
    :param yticks: y-axis ticks
    :param xlabel_size: x-axis label size
    :param ylabel_size: y-axis label size
    :param title_str: title
    :param xticks: x-axis ticks
    :param yticks: y-axis ticks
    :param xlabel_size: x-axis label size
    :param ylabel_size: y-axis label size
    :param title_size: title size
    :param ticks_size: ticks size
    :param legend_size: legend size
    :param legend_loc: legend location
    :return: None.
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if xlabel_size:
        ax.xaxis.label.set_size(xlabel_size)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylabel_size:
        ax.yaxis.label.set_size(ylabel_size)
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if ticks_size:
        ax.tick_params(axis="both", which="major", labelsize=ticks_size)
    if legend:
        ax.legend(legend, fontsize=legend_size, loc=legend_loc)
    if title_str:
        ax.set_title(title_str, fontsize=title_size)

def set_plot_properties(
    font_size: float = 20,
    legend_font_size: float = 14,
    xtick_label_size: float = 14,
    ytick_label_size: float = 14,
    markersize: float = 10,
    ) -> None:
    """Sets plot properties.

    :param font_size: font size
    :param legend_font_size: legend font size
    :param xtick_label_size: xtick label size
    :param ytick_label_size: ytick label size
    :param markersize: marker size
    :param usetex: use tex
    :return: None.
    """
    sns.set_color_codes()
    sns.set()
    font = {"family": "normal", "weight": "bold", "size": font_size}
    plt.rc("font", **font)
    plt.rcParams["text.latex.preamble"] = r"\boldmath"

    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.weight"] = "bold"

    params = {
        "legend.fontsize": legend_font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": xtick_label_size,
        "ytick.labelsize": ytick_label_size,
        "lines.markersize": markersize,
        "figure.autolayout": True,
    }

    pylab.rcParams.update(params)

    sns.set_style(style="darkgrid")
    
def plot_stacked_barplot(
    df=None,
    x_var=None,
    y_var=None,
    ylim=None,
    title_str=None,
    pal=None,
    ax=None,
    y_label=None,
    ) -> None:
    """Plots a grouped barplot.

    :param df: dataframe
    :param x_var: x-axis variable
    :param y_var: y-axis variable
    :param ylim: y-axis limits
    :param title_str: title of the plot
    :param pal: palette
    :param ax: axis to plot on
    :return: None.
    """
    if not pal:
        df.plot(kind="bar", stacked=True, ax=ax, x=x_var, y=y_var)

    if pal:
        df.plot(
            kind="bar", stacked=True, ax=ax, x=x_var, y=y_var, color=pal
        )

    ### set y label
    if y_label:
        ax.set_ylabel(y_label)

    # Set axis infos
    set_axis_infos(ax, ylim=ylim, title_str=title_str)

def plot_stacked_grouped_barplot(
    df=None,
    groups=None,
    ylim=None,
    title_str=None,
    pal=None,
    ax=None,
    y_label=None,
    ) -> None:
    """Plots a grouped barplot.

    :param df: dataframe
    :param x_var: x-axis variable
    :param y_var: y-axis variable
    :param ylim: y-axis limits
    :param title_str: title of the plot
    :param pal: palette
    :param ax: axis to plot on
    :return: None.
    """

    wd = 1/(len(groups)+1)
    if not pal:
           for g in range(len(groups)):
            df[groups[g]].plot(kind='bar', stacked=True, ax=ax, width=wd, position=len(groups)-g, label="Group "+str(len(groups)-g))
    if pal:
        colors = [pal[len(groups[0])*g:len(groups[0])*(g+1)] for g in range(len(groups))]
        for g in range(len(groups)):
                df[groups[g]].plot(kind='bar', stacked=True, ax=ax, width=wd, position=len(groups)-g, label="Group "+str(len(groups)-g), color=colors[g])
     ### set y label
    if y_label:
        ax.set_ylabel(y_label)

    # Set axis infos
    set_axis_infos(ax, ylim=ylim, title_str=title_str)
    
def plot_bar_DL_old(rho_initt, alpha_initt, rho_alloc, alpha_alloc, ylabell):
    # Example Plots location
    _X1_DATA = np.reshape(rho_initt, [-1,1]).squeeze()
    _X2_DATA = np.reshape(alpha_initt, [-1,1]).squeeze()
    _X3_DATA = np.reshape(rho_alloc, [-1,1]).squeeze()
    _X4_DATA = np.reshape(alpha_alloc, [-1,1]).squeeze()

    # Concatenate the data
    _X_DATA = [_X1_DATA, _X2_DATA, _X3_DATA, _X4_DATA]
    _X_LABEL = ["$r_%d$" % i for i in range(np.shape(rho_initt)[0])]
    _DATA_FRAME = pd.DataFrame(
        {"Localrho": _X1_DATA,"Localalpha": _X2_DATA, "Allocatedrho": _X3_DATA, "Allocatedalpha": _X4_DATA, "Edge Robot ID": _X_LABEL}
    )
    set_plot_properties()
    df = _DATA_FRAME
    x_var = "Edge Robot ID"
    y_var = ["Localrho","Localalpha", "Allocatedrho", "Allocatedalpha"]
    y_label = ylabell
    fig, ax = plt.subplots(figsize=(8, 8))
    # Create a stacked boxplot
    plot_stacked_barplot(
        df=df,
        x_var=x_var,
        y_var=y_var,
        title_str="",
        ax=ax,
        y_label=y_label,
    )
    ax.legend().remove()
    return ax, fig

def plot_bar_DL(rho_initt, alpha_initt, rho_alloc, alpha_alloc, ylabell):
    # Example Plots location
    _X1_DATA = np.reshape(rho_initt, [-1,1]).squeeze()
    _X2_DATA = np.reshape(alpha_initt, [-1,1]).squeeze()
    _X3_DATA = np.reshape(rho_alloc, [-1,1]).squeeze()
    _X4_DATA = np.reshape(alpha_alloc, [-1,1]).squeeze()

    # Concatenate the data
    _X_DATA = [_X1_DATA, _X2_DATA, _X3_DATA, _X4_DATA]
    _X_LABEL = ["$r_%d$" % i for i in range(np.shape(rho_initt)[0])]
    _DATA_FRAME = pd.DataFrame(
        {"Localrho": _X1_DATA,"Localalpha": _X2_DATA, "Allocatedrho": _X3_DATA, "Allocatedalpha": _X4_DATA, "Edge Robot ID": _X_LABEL}
    )
    _GROUPED_DATA = [["Localrho","Allocatedrho"],["Localalpha","Allocatedalpha"]]
    _DATA_FRAME.set_index('Edge Robot ID', inplace=True)   
    set_plot_properties()
    df = _DATA_FRAME
    y_label = ylabell
    fig, ax = plt.subplots(figsize=(8, 8))
    # Create a stacked boxplot
    plot_stacked_grouped_barplot(
        df=df,
        groups=_GROUPED_DATA,
        title_str="",
        ax=ax,
        y_label=y_label,
        pal = sns.color_palette("Paired")
    )
    ax.legend().remove()
    return ax, fig
def plot_util_DL(nominal, rho_contrib, alpha_contrib, ylabell):
    # Example Plots location
    _X1_DATA = np.reshape(nominal, [-1,1]).squeeze()
    _X2_DATA = np.reshape(rho_contrib, [-1,1]).squeeze()
    _X3_DATA = np.reshape(alpha_contrib, [-1,1]).squeeze()

    # Concatenate the data
    _X_DATA = [_X1_DATA, _X2_DATA, _X3_DATA]
    _X_LABEL = ["$r_%d$" % i for i in range(np.shape(nominal)[0])]
    _DATA_FRAME = pd.DataFrame(
        {"nominal": _X1_DATA,"rho_contrib": _X2_DATA, "alpha_contrib": _X3_DATA, "Edge Robot ID": _X_LABEL}
    )
    set_plot_properties()
    df = _DATA_FRAME
    x_var = "Edge Robot ID"
    y_var = ["nominal","rho_contrib", "alpha_contrib"]
    y_label = ylabell
    fig, ax = plt.subplots(figsize=(8, 8))
    # adjust color palette
    custom_palette = sns.color_palette("Paired")

    # Adjust the palette to start from the specified color
    adjusted_palette = custom_palette[8:10] + custom_palette[5:6]
    # Create a stacked boxplot
    plot_stacked_barplot(
        df=df,
        x_var=x_var,
        y_var=y_var,
        title_str="",
        ax=ax,
        y_label=y_label,
        pal=adjusted_palette
    )
    ax.legend().remove()
    return ax, fig

def plot_barr(Ee, fs_alloc):
    # Example Plots location
    _X1_DATA = np.reshape(Ee, [-1,1]).squeeze()
    _X2_DATA = np.reshape(fs_alloc, [-1,1]).squeeze()

    # Concatenate the data
    _X_DATA = [_X1_DATA, _X2_DATA]
    _X_LABEL = ["$r_%d$" % i for i in range(np.shape(Ee)[0])] # DONE
    _DATA_FRAME = pd.DataFrame(
        {"Local": _X1_DATA, "Allocated": _X2_DATA, "Edge Robot ID": _X_LABEL}
    )
    set_plot_properties()
    df = _DATA_FRAME
    x_var = "Edge Robot ID"
    y_var = ["Local", "Allocated"]
    y_label = "Early Exit ID"
    fig, ax = plt.subplots(figsize=(8, 8))
    # Create a stacked boxplot
    plot_stacked_barplot(
        df=df,
        x_var=x_var,
        y_var=y_var,
        title_str="",
        ax=ax,
        y_label=y_label,
        pal=sns.color_palette("Paired")
    )
    ax.legend().remove()
    return ax, fig
def plot_util(Ee, fs_alloc):
    # Example Plots location
    _X1_DATA = np.reshape(Ee, [-1,1]).squeeze()
    _X2_DATA = np.reshape(fs_alloc, [-1,1]).squeeze()

    # Concatenate the data
    _X_DATA = [_X1_DATA, _X2_DATA]
    _X_LABEL = ["$r_%d$" % i for i in range(np.shape(Ee)[0])] # DONE
    _DATA_FRAME = pd.DataFrame(
        {"Nominal": _X1_DATA, "Acquired": _X2_DATA, "Edge Robot ID": _X_LABEL}
    )
    set_plot_properties()
    df = _DATA_FRAME
    x_var = "Edge Robot ID"
    y_var = ["Nominal", "Acquired"]
    y_label = "Model Accuracy"
    fig, ax = plt.subplots(figsize=(8, 8))
    custom_palette = sns.color_palette("Paired")

    # Adjust the palette to start from the specified color
    adjusted_palette = custom_palette[8:10]
    # Create a stacked boxplot
    plot_stacked_barplot(
        df=df,
        x_var=x_var,
        y_var=y_var,
        title_str="",
        ax=ax,
        y_label=y_label,
        pal=adjusted_palette
    )
    ax.legend().remove()
    return ax, fig

def plot_paired_boxplot(
    df=None,
    x_var=None,
    y_var=None,
    ylim=None,
    title_str=None,
    order_list=None,
    pal=None,
    hue=None,
    ax=None,
    whiss = None,
    ) -> None:
    """Plots a paired boxplot.

    :param df: dataframe
    :param x_var: x-axis variable
    :param y_var: y-axis variable
    :param ylim: y-axis limits
    :param title_str: title of the plot
    :param order_list: order of the x-axis variable
    :param pal: palette
    :param hue: hue variable
    :param ax: axis to plot on
    :return: None.
    """
    # Plots a boxplot
    if not pal:
        # Plots a boxplot with order
        if order_list:
            sns.boxplot(
                x=x_var, y=y_var, data=df, order=order_list, hue=hue, ax=ax
            )
        else:
            sns.boxplot(
                x=x_var, y=y_var, data=df, order=order_list, hue=hue, ax=ax, whis=whiss
            )

    # Plots a boxplot with palette
    if pal:
        if order_list:
            sns.boxplot(
                x=x_var,
                y=y_var,
                data=df,
                order=order_list,
                palette=pal,
                hue=hue,
                ax=ax,
            )
            
        else:
            sns.boxplot(
                x=x_var,
                y=y_var,
                data=df,
                order=order_list,
                palette=pal,
                hue=hue,
                ax=ax,
                whis=whiss,
            )

    # Set axis infos
    set_axis_infos(ax, ylim=ylim, title_str=title_str)
