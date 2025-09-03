# Refactor All

## Refactor BumpChart

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import plotly.graph_objects as go
from matplotlib.colors import to_rgb
from typing import Optional

def custom_bumpchart(df, ax, smooth=9, n=100, title="Ranking Over Time"):
    """
    Creates a smoothed bump chart on a given matplotlib axes object.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame where index = categories and columns = time points.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object on which to draw.
    smooth : int, optional
        Steepness of sigmoid smoothing for the line curves.
    n : int, optional
        Number of interpolated points between each pair of observations.
    title : str, optional
        Chart title.
    """
    # --- Helper functions for smoothing ---
    def _sigmoid(xs, ys, smooth=8, n=100):
        (x_from, x_to), (y_from, y_to) = xs, ys
        xs_interp = np.linspace(-smooth, smooth, num=n)[:, None]
        ys_interp = np.exp(xs_interp) / (np.exp(xs_interp) + 1)
        return (
            ((xs_interp + smooth) / (smooth * 2) * (x_to - x_from) + x_from),
            (ys_interp * (y_to - y_from) + y_from)
        )

    def _sigmoid_pairwise(xs, ys, smooth=8, n=100):
        xs_pairs = np.lib.stride_tricks.sliding_window_view(xs, 2)
        ys_pairs = np.lib.stride_tricks.sliding_window_view(ys, 2)
        interp_x, interp_y = _sigmoid(xs_pairs.T, ys_pairs.T, smooth=smooth, n=n)
        return interp_x.T.flatten(), interp_y.T.flatten()

    # --- 1. Long-format data ---
    long_df = (
        df.reset_index()
          .melt(id_vars=df.index.name, var_name="time", value_name="value")
    )

    # --- 2. Plot lines for each category ---
    for category, group in long_df.groupby(df.index.name):
        group = group.sort_values('time')
        xs, ys = np.arange(len(group['time'])), group['value'].values

        # smooth curve
        interp_x, interp_y = _sigmoid_pairwise(xs, ys, smooth=smooth, n=n)

        # plot
        line, = ax.plot(interp_x, interp_y, lw=3)
        ax.scatter(xs, ys, s=100, color=line.get_color(), zorder=5)

        # annotate at last point
        ax.annotate(
            f"{str(category).title()}",
            xy=(interp_x[-1], interp_y[-1]),
            xytext=(5, 0), textcoords="offset points",
            color=line.get_color(), va="center", fontweight="bold"
        )

    # --- 3. Axis styling ---
    ax.set_frame_on(False)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylabel("Rank")
    ax.margins(x=0.05, y=0.05)
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------
if __name__ == "__main__":
    categories = [f"Team {i+1}" for i in range(5)]
    data = {
        2016: [1, 5, 3, 2, 4],
        2017: [1, 2, 5, 3, 4],
        2018: [1, 3, 4, 2, 5],
        2019: [2, 1, 5, 3, 4],
        2020: [1, 3, 5, 2, 4],
        2021: [2, 1, 5, 3, 4],
        2022: [1, 2, 4, 3, 5],
        2023: [2, 1, 3, 5, 4],
    }
    df = pd.DataFrame(data, index=categories)
    df.index.name = "category"

    fig, ax = plt.subplots(figsize=(12, 6))
    custom_bumpchart(df, ax, title="Custom Bump Chart")
    fig.tight_layout()
    plt.show()
## Refactor Stack

def custom_stackedbar(
    df: pd.DataFrame,
    ax: plt.Axes,
    *,
    cmap: str = "tab10",
    bar_width: float = 0.6,
    legend: bool = True,
    legend_ncols: int = 2,
    legend_fontsize: int = 10,
    legend_edgecolor: str = "#444444",
    show_frame: bool = False
):
    """
    Render a stacked bar chart from a wide-format dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format data. Each row = category (bars), each column = subcategory stacked segments.
    ax : matplotlib.axes.Axes
        Target axes to draw onto.

    Keyword Args
    ------------
    cmap : str
        Colormap name or custom colormap for stacked colors.
    bar_width : float
        Width of each bar.
    legend : bool
        If True, show legend.
    legend_ncols : int
        Number of columns in legend.
    legend_fontsize : int
        Font size for legend.
    legend_edgecolor : str
        Edge color for legend box.
    show_frame : bool
        If True, keep the axes frame visible.
    """
    # ensure copy so original df not touched
    df_data = df.copy()
    categories = df_data.columns
    row_values = {idx: df_data.loc[idx].values for idx in df_data.index}

    # palette
    cmap_obj = plt.get_cmap(cmap, len(row_values))

    # stacking logic
    bottom = np.zeros(len(categories))
    for i, (row, values) in enumerate(row_values.items()):
        ax.bar(
            categories,
            values,
            width=bar_width,
            label=row,
            color=cmap_obj(i),
            bottom=bottom
        )
        bottom += values

    # legend
    if legend:
        ax.legend(
            loc="best",
            ncols=legend_ncols,
            fontsize=legend_fontsize,
            edgecolor=legend_edgecolor
        )

    # frame toggle
    ax.set_frame_on(show_frame)
    ax.set_axisbelow(True)
    # evenly spaced ticks
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(50))  # every 200
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))  # integer format
    return ax


# --------------------------- Example Usage ---------------------------
if __name__ == "__main__":
    species = ("Adelie", "Chinstrap", "Gentoo", "Others")
    weight_counts = {
        "Cat1": np.array([70, 31, 58, 20]),
        "Cat2": np.array([82, 37, 66, 21]),
        "Cat3": np.array([21, 14, 10, 22]),
        "Cat4": np.array([10, 5, 6, 33]),
        "Cat5": np.array([11, 2, 3, 8]),
        "Cat6": np.array([1, 3, 5, 6]),
    }
    df_species = pd.DataFrame(weight_counts, index=species).T

    fig, ax = plt.subplots(figsize=(9, 4))
    custom_stackedbar(df_species, ax, cmap="cust_blue_grad", bar_width=0.7, show_frame=False)
    plt.tight_layout()
    plt.show()
## Refactor Sankey


def custom_sankey(
    df_links: pd.DataFrame,
    df_nodes: pd.DataFrame,
    *,
    font_family: str = "Poppins",
    font_size: int = 14,
    font_color: str = "#BAB9B9",
    paper_bgcolor: str = "#282c34",
    plot_bgcolor: str = "#282c34",
    node_color_default: str = "#678AA2",
    link_color_default: str = "#404040",
    title: str = "",
    width: int = 900,
    height: int = 500
):
    """
    Create a formatted Sankey diagram from link/node dataframes.

    Parameters
    ----------
    df_links : pd.DataFrame
        Must contain 'source', 'target', 'values'. Optionally 'color'.
    df_nodes : pd.DataFrame
        Must contain 'node', 'values'. Optionally 'color'.
    font_family, font_size, font_color : str/int
        Font configuration.
    paper_bgcolor, plot_bgcolor : str
        Background colors.
    node_color_default, link_color_default : str
        Fallback colors if df doesn't specify.
    title : str
        Plot title.
    width, height : int
        Figure size.
    """

    # ---- Helper to format values nicely ----
    def _best_format(value):
        if value >= 10**9:
            return f'{value/1e9:.1f} Bn'
        if value >= 10**6:
            return f'{value/1e6:.1f} Mn'
        if value >= 10**3:
            return f'{value/1e3:.1f} K'
        return f'{value:.1f}'

    # ---- Prepare nodes ----
    df_nodes = df_nodes.copy()
    if "color" not in df_nodes:
        df_nodes["color"] = node_color_default
    df_nodes["values_x"] = df_nodes["values"].apply(_best_format)
    df_nodes["label_raw"] = (
        df_nodes["node"].astype(str) + "\n" + df_nodes["values_x"].astype(str)
    )

    def _format_label(label):
        parts = label.split("\n")
        if len(parts) == 2:
            return f"<b>{parts[0]}</b><br>{parts[1]}"
        return f"<b>{label}</b>"

    node_labels = [_format_label(lbl) for lbl in df_nodes["label_raw"].values]

    # ---- Prepare links ----
    df_links = df_links.copy()
    if "color" not in df_links:
        df_links["color"] = link_color_default

    links = dict(
        source=df_links["source"].values,
        target=df_links["target"].values,
        value=df_links["values"].values,
        color=df_links["color"].values,
    )

    # ---- Build Sankey ----
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=5,
                thickness=10,
                line=dict(color="#5da3f9", width=0.1),
                color=df_nodes["color"].values,
                label=node_labels,
            ),
            link=links,
        )
    )

    # ---- Layout ----
    fig.update_layout(
        title_text=title,
        font=dict(family=font_family, size=font_size, color=font_color),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        width=width,
        height=height,
    )

    return fig


# ----------------- Example usage -----------------
if __name__ == "__main__":
    # Load data
    df_sankey_1 = pd.read_csv("flatfile/sankey_1.csv", delimiter="|")
    df_sankey_node_1 = pd.read_csv("flatfile/sankey_node_1.csv", delimiter="|")

    # Call custom function
    fig = custom_sankey(df_links=df_sankey_1, df_nodes=df_sankey_node_1, title="My Sankey Flow")
    fig.show()

# --------------------
# Contrast helpers
# --------------------
def _rel_lum(rgb):
    """Relative luminance of an RGB color."""
    srgb = np.asarray(rgb)
    lin  = np.where(srgb <= 0.03928,
                    srgb / 12.92,
                    ((srgb + 0.055) / 1.055) ** 2.4)
    return np.dot(lin, [0.2126, 0.7152, 0.0722])

def _contrast(a, b):
    """WCAG contrast ratio."""
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)

def _choose_fg(rgba_bg, dark="#222222", light="#FAFAFA", target=4.5):
    """Pick dark/light text for a background that meets contrast target."""
    L_bg = _rel_lum(rgba_bg[:3])
    for cand in (dark, light):
        if _contrast(L_bg, _rel_lum(to_rgb(cand))) >= target:
            return cand
    return max((dark, light), key=lambda c: _contrast(L_bg, _rel_lum(to_rgb(c))))


# --------------------
# Reusable Heatmap
# --------------------

def custom_heatmap(
    data,
    row_labels,
    col_labels,
    *,
    ax=None,
    cmap="viridis",
    edgecolor="#282c34",
    linewidth=2,
    cbarlabel="",
    cbar_kw=None,
    annotate=False,
    fmt="smart",
    fontsize=11
):
    """
    Create a heatmap with optional annotation and smart formatting.

    Parameters
    ----------
    data : 2D ndarray
        Matrix of values to plot.
    row_labels, col_labels : list of str
        Labels for the rows/columns.
    ax : matplotlib.axes.Axes, optional
        Target axis (creates new one if None).
    cmap : str
        Colormap.
    edgecolor : str
        Color of grid cell edges.
    linewidth : float
        Edge line width.
    cbarlabel : str
        Label for the colorbar.
    cbar_kw : dict
        Extra kwargs passed to colorbar.
    annotate : bool
        If True, show values inside cells.
    fmt : str
        If "smart", formats as K/Mn/Bn, otherwise uses Python format string.
    fontsize : int
        Font size of annotations.
    """
    if ax is None:
        ax = plt.gca()
    if cbar_kw is None:
        cbar_kw = {}

    # grid edges
    x = np.arange(data.shape[1] + 1)
    y = np.arange(data.shape[0] + 1)

    im = ax.pcolormesh(x, y, data, cmap=cmap,
                       edgecolors=edgecolor, linewidth=linewidth)

    # colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.ax.set_frame_on(False)

    # # Format colorbar tick labels using best_format only if fmt == "smart"
    # def best_format(value):
    #     try:
    #         num = float(value)
    #     except (TypeError, ValueError):
    #         return value
    #     if abs(num) >= 1_000_000_000:
    #         return f"{num / 1_000_000_000:.1f} Bn"
    #     elif abs(num) >= 1_000_000:
    #         return f"{num / 1_000_000:.1f} Mn"
    #     elif abs(num) >= 1_000:
    #         return f"{num / 1_000:.1f} K"
    #     else:
    #         return f"{num:.1f}"

    # if fmt == "smart":
    #     # Use y-axis for vertical colorbar, x-axis for horizontal
    #     if getattr(cbar, "orientation", "vertical") == "horizontal":
    #         tick_locs = cbar.ax.get_xticks()
    #         cbar.ax.set_xticks(tick_locs)
    #         cbar.ax.set_xticklabels([best_format(t) for t in tick_locs])
    #     else:
    #         tick_locs = cbar.ax.get_yticks()
    #         cbar.ax.set_yticks(tick_locs)
    #         cbar.ax.set_yticklabels([best_format(t) for t in tick_locs])

    # # Fix for bar color smaller than ticker: set aspect ratio to 'auto'
    # ax.set_aspect('auto')

    # ticks
    ax.set_xticks(np.arange(data.shape[1]) + 0.5)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.grid(False)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    texts = []
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if fmt == "smart":
                    txt_val = (
                        f"{val/1e9:.1f} Bn" if val >= 1e9 else
                        f"{val/1e6:.1f} Mn" if val >= 1e6 else
                        f"{val/1e3:.1f} K" if val >= 1e3 else
                        f"{val:.1f}"
                    )
                else:
                    txt_val = format(val, fmt)

                rgba = im.cmap(im.norm(val))
                colour = _choose_fg(rgba)
                txt = ax.text(j + 0.5, i + 0.5, txt_val,
                              ha="center", va="center",
                              color=colour, fontsize=fontsize)
                texts.append(txt)

    return im, cbar, texts

# -------------------- Example --------------------
if __name__ == "__main__":
    data = np.random.uniform(10_000, 6_000_000_000, size=(6, 7))
    fig, ax = plt.subplots(figsize=(10, 5))

    custom_heatmap(
        data,
        [f"R{i}" for i in range(6)],
        [f"C{i}" for i in range(7)],
        ax=ax,
        annotate=True,
        cbarlabel="Probability",
        cmap="cust_black_grad",
        linewidth=4,
        edgecolor=themes['dark']['background']
    )

    plt.tight_layout()
    plt.show()

## Refactor GroupBar


def custom_groupedbar(
    df: pd.DataFrame,
    ax: plt.Axes,
    *,
    width: float = 0.2,
    cmap: str = "tab10",
    ylabel: str = "",
    title: str = "",
    legend_loc: str = "upper left",
    legend_ncols: int = 2,
    legend_fontsize: int = 10,
    legend_edgecolor: str = "#444444",
    show_labels: bool = True,
):
    """
    Create a grouped bar chart from a wide-format dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format data. Rows = attributes, columns = categories.
    ax : matplotlib.axes.Axes
        Axis object to draw on.
    width : float
        Bar width for each group.
    cmap : str
        Matplotlib colormap name.
    ylabel : str
        Y-axis label.
    title : str
        Chart title.
    legend_loc : str
        Legend location.
    legend_ncols : int
        Legend number of columns.
    legend_fontsize : int
        Legend font size.
    legend_edgecolor : str
        Legend box edge color.
    show_labels : bool
        Annotate each bar with value.
    """
    categories = df.columns.values
    values = {idx: df.loc[idx].values for idx in df.index}

    x = np.arange(len(categories))  # category positions
    n_groups = len(values)
    cmap_obj = plt.get_cmap(cmap, n_groups)

    # plotting loop
    for i, (attribute, measurement) in enumerate(values.items()):
        offset = width * i
        color = cmap_obj(i)
        rects = ax.bar(
            x + offset,
            measurement,
            width,
            color=color,
            label=attribute,
            alpha=1,
        )
        if show_labels:
            def best_format(value):
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    return value
                if abs(num) >= 1_000_000_000:
                    return f"{num / 1_000_000_000:.1f} Bn"
                elif abs(num) >= 1_000_000:
                    return f"{num / 1_000_000:.1f} Mn"
                elif abs(num) >= 1_000:
                    return f"{num / 1_000:.1f} K"
                else:
                    return f"{abs(num):.1f}"
            ax.bar_label(rects, labels=[best_format(v) for v in measurement], padding=3)

    # xtick centers under grouped bars
    group_centers = x + width * (n_groups / 2 - 0.5)
    ax.set_xticks(group_centers, categories)

    # formatting
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.legend(
        loc=legend_loc,
        ncols=legend_ncols,
        edgecolor=legend_edgecolor,
        fontsize=legend_fontsize,
    )
    ax.set_axisbelow(True)
    ax.set_frame_on(False)

    return ax


# -------------------- Example --------------------
if __name__ == "__main__":
    # Data Preparation
    categories = ("Adelie", "Chinstrap", "Gentoo")
    values = {
        "Bill Depth": (18.3, 18.4, 14.9),
        "Bill Length": (38.7, 48.8, 47.5),
        "Flipper Depth": (90.2, 90.2, 11.3),
        "Flipper Length": (189.9, 195.8, 217.1),
    }
    df_data = pd.DataFrame(values).T
    df_data.columns = categories

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    custom_groupedbar(
        df_data,
        ax,
        width=0.2,
        cmap="cust_black_grad",
        ylabel="Length (mm)",
        title="Penguin attributes by species",
        legend_loc="upper left",
        legend_ncols=4
    )
    plt.show()

## Refactor Bar
def custom_bar(categories, values, ax, cmap="tab10"):
    """
    Plot a vertical bar chart on the given axes.

    Parameters
    ----------
    categories : list
        Category labels for the bars.
    values : list
        Heights of the bars.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    cmap : str or Colormap, optional
        Colormap name or Colormap object for bar colors.
    """
    def best_format(value):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return value  # non-numeric → pass through unchanged

        if abs(num) >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f} Bn"
        elif abs(num) >= 1_000_000:
            return f"{num / 1_000_000:.1f} Mn"
        elif abs(num) >= 1_000:
            return f"{num / 1_000:.1f} K"
        else:
            return f"{num:.1f}"

    # Get colors from colormap
    cmap_obj = plt.get_cmap(cmap, len(categories))
    bar_colors = [cmap_obj(i) for i in range(len(categories))]
    bars = ax.bar(x=categories, height=values, color=bar_colors)
    ax.bar_label(container=bars, labels=[best_format(v) for v in values], padding=10)
    ax.set(xlabel='Values', ylabel='Categories')
    ax.set_axisbelow(True)
    ax.set_frame_on(False)
    return ax


if __name__ == "__main__":

    categories = ["A", "B", "C", "D"]
    values = [500, 2500, 12000, 3_200_000]

    fig, ax = plt.subplots(figsize=(6, 4))
    custom_bar(categories, values, ax)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()
## Refactor Histogram


def custom_histogram(
    data,
    ax,
    bins: int = 30,
    density: bool = True,
    color: str = "C0",
    alpha: float = 0.3,
    xlabel: str = "",
    ylabel: str = "Density",
    title: str = "",
    outline_width: float = 1.5,
    outline_color: Optional[str] = None,
):
    """
    Plot a styled histogram with both filled and outlined layers.

    Parameters
    ----------
    data : array-like
        Input values.
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    bins : int
        Number of histogram bins.
    density : bool
        Normalize histogram (probability density).
    color : str
        Fill color for histogram bars.
    alpha : float
        Transparency of the filled histogram.
    xlabel, ylabel, title : str
        Labels and title for the plot.
    outline_width : float
        Line width for the histogram outline.
    outline_color : str or None
        Color for the outline (defaults to same as fill color).
    """
    if outline_color is None:
        outline_color = color

    # filled histogram (no edges)
    ax.hist( data, bins=bins, density=density, histtype="stepfilled", facecolor=color, alpha=alpha, edgecolor="none",)
    # outlined histogram (always on top)
    ax.hist( data, bins=bins, density=density, histtype="step", color=outline_color, linewidth=outline_width, alpha=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.set_frame_on(False)
    return ax


# ----------------- Example usage -----------------
if __name__ == "__main__":
    # Generate random normal data
    mu_x, sigma_x = 200, 25
    x = np.random.normal(mu_x, sigma_x, size=1000)

    set_theme('dark', custom_font=custom_font, custom_colors=custom_colors)
    fig, ax = plt.subplots(figsize=(10, 6))

    custom_histogram(
        x,
        ax=ax,
        bins=90,
        color=custom_colors[0],
        alpha=0.3,
        xlabel="Value",
        ylabel="Density",
        title="Distribution of X",
        outline_width=1.5
    )

    fig.tight_layout()
    plt.show()
## Refactor Circular


def custom_circularbar(
    percentages_dict: dict,
    *,
    ax=None,
    cmap: str = "tab20",
    bar_width: float = 0.4,
    title: str = "Circular Bar Plot",
    figsize=(5, 5)
):
    """
    Create a circular bar chart from a dictionary of category percentages.

    Parameters
    ----------
    percentages_dict : dict
        Dictionary {label: value}.
    ax : matplotlib.axes.Axes, optional
        Polar axis to draw on (creates one if None).
    cmap : str
        Colormap for bar colors.
    bar_width : float
        Angular width of each bar (radians).
    title : str
        Chart title.
    figsize : tuple
        Figure size if ax is None.

    Returns
    -------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        The polar axis with the chart.
    """
    # --- Prepare data ---
    sorted_labels = sorted(percentages_dict.keys())
    sorted_values = [percentages_dict[label] for label in sorted_labels]
    N = len(sorted_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    cmap_obj = plt.get_cmap(cmap, N)
    colors = [cmap_obj(i) for i in range(N)]

    # --- Create polar axis if not provided ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # polar settings
    ax.set_theta_offset(np.pi / 2)     # start from top
    ax.set_theta_direction(-1)         # clockwise
    ax.set_axisbelow(True)

    # --- Bars ---
    bars = ax.bar(angles, sorted_values, width=bar_width, color=colors)

    # --- Clean up chart ---
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_frame_on(False)

    # --- Title ---
    ax.set_title(title, fontsize=12)

    return ax

if __name__ == "__main__":

    percentages_dict_expenses = {
        "Apple": 25,
        "Banana": 28,
        "Orange": 22,
        "Mango": 18,
        "Grapes": 15,
        "Strawberry": 10,
        "Pineapple": 8
    }

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    custom_circularbar(percentages_dict_expenses, ax=ax, cmap="plasma", title="Category Percentages")
    plt.tight_layout()
    plt.show()
## Refactor Waterfall

def plot_waterfall_horizontal(
    categories, values, ax,
    *,
    title="Waterfall Chart (Horizontal)",
    green="#219d55", red="#e74c3c", blue="#3498db"
):
    """
    Creates a horizontal waterfall chart with TOTAL at the bottom.

    Parameters
    ----------
    categories : list of str
        Labels for each step (excluding total).
    values : list of float
        Numeric changes (+ for gain, - for loss).
    ax : matplotlib.axes.Axes
        Axis object to plot on.
    title : str
        Title of the chart.
    green, red, blue : str
        Colors for positive, negative, and total bars.
    """
    total = sum(values)
    cum = np.cumsum([0] + values[:-1])  

    # assign colors
    colors = [green if v > 0 else red for v in values]
    colors = [blue] + colors   # add total color at front

    # insert total at bottom
    categories = ["Total"] + categories
    values = [total] + values
    cum = [0] + list(cum)  # shift cumulative positions

    # --- plot bars ---
    for i, (c, v, color) in enumerate(zip(cum, values, colors)):
        if i == 0:  # total bar
            ax.barh(i, v, left=0, color=color)
            ax.text(v/2, i, f"{v:+.1f}", va="center", ha="center",
                    color="white", fontsize=12)
        else:       # incremental bars
            ax.barh(i, v, left=c, color=color)
            ax.text(c + v/2, i, f"{v:+.1f}", va="center", ha="center",
                    color="white", fontsize=12)

    # --- styling ---
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xticks([])
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.axvline(total, color="#969696", linestyle="--", linewidth=1, zorder=0)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_frame_on(False)

    return ax

if __name__ == "__main__":
    categories = ["Postpaid Pack", "Postpaid PayU", "Others",
                  "AddON", "Prepaid Pack", "Prepaid Recovery"][::-1]
    values = [-3.0, -2.0, +10, -1.2, 2, +2.5][::-1]  # in Bn

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_waterfall_horizontal(categories, values, ax,
                              title="Revenue Drivers (Bn MoM)")
    plt.tight_layout()
    plt.show()

from matplotlib.patches import FancyArrowPatch

def custom_bar_arrow(categories, values, ax, cmap="tab10"):
    """
    Plot a vertical bar chart on the given axes.

    Parameters
    ----------
    categories : list
        Category labels for the bars.
    values : list
        Heights of the bars.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    cmap : str or Colormap, optional
        Colormap name or Colormap object for bar colors.
    """

    green, red = '#06b45e', '#fc3d51'
    def best_format(value):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return value  # non-numeric → pass through unchanged

        if abs(num) >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f} Bn"
        elif abs(num) >= 1_000_000:
            return f"{num / 1_000_000:.1f} Mn"
        elif abs(num) >= 1_000:
            return f"{num / 1_000:.1f} K"
        elif abs(num) >= 1:
            return f"{num:.1f}"
        else:
            return f"{num:.3f}"

    # Get colors from colormap
    cmap_obj = plt.get_cmap(cmap, len(categories))
    bar_colors = [cmap_obj(i) for i in range(len(categories))]
    bars = ax.bar(x=categories, height=values, color=bar_colors)
    ax.bar_label(container=bars, labels=[best_format(v) for v in values], padding=10)
    ymax = max(values) * 2
    ax.set(yticks=[], ylim=(0, ymax))
    ax.set_axisbelow(True)
    ax.set_frame_on(True)
    mid_x = (bars[0].get_x() + bars[0].get_width()/2 + bars[1].get_x() + bars[1].get_width()/2) / 2
    mid_y = max(values) * 1.7
    pct_change = (values[1] / values[0] - 1) * 100
    pct_text = f"{pct_change:+.1f}%"
    bbox_color = green if pct_change >= 0 else red

    # Annotate with background color
    annotation = ax.annotate(
        pct_text,
        xy=(mid_x, mid_y),
        xytext=(0, 0),
        textcoords='offset points',
        ha='center', va='center',
        fontsize=10,
        color='white',
        bbox=dict(boxstyle="round,pad=0.3", fc=bbox_color, ec='none', alpha=0.85)
    )

    # Define arrow start and end points
    # Arrow from top of first bar to top of second bar
    bar0 = bars[0]
    bar1 = bars[1]
    x0 = bar0.get_x() + bar0.get_width() / 2
    y0 = bar0.get_height()
    x1 = bar1.get_x() + bar1.get_width() / 2
    y1 = bar1.get_height()
    # Draw two arrows: from bar0 to mid, and from mid to bar1
    arrow1 = FancyArrowPatch(
        posA=(x0, y0), posB=(mid_x, mid_y),
        arrowstyle="-|>", lw=1.5, color=bbox_color, alpha=0.5,
        connectionstyle="angle,angleA=90,angleB=0"
    )
    arrow2 = FancyArrowPatch(
        posA=(mid_x, mid_y), posB=(x1, y1),
        arrowstyle="-|>", lw=1.5, color=bbox_color, alpha=0.5,
        connectionstyle="angle,angleA=0,angleB=90"
    )
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

    return ax


if __name__ == "__main__":

    categories = ["m0", "m1"]
    values = [9300, 9359]

    fig, ax = plt.subplots(figsize=(2, 2))
    custom_bar_arrow(categories, values, ax, cmap='cust_blue_grad')
    plt.tight_layout()
    plt.show()