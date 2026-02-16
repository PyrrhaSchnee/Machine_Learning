#!/usr/bin/env python3
"""
Histogram.py

Visualize the "Test_knight.csv" via histograms
Then visualize the superposition of the "Test_knight.csv"
and the "Train_knight.csv" via histograms
All the histograms stay in one single big figure (subplot)
One histogram per feature
"""

from __future__ import annotations
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST_FILE: str = "Test_knight.csv"
TRAIN_FILE: str = "Train_knight.csv"


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Take a string or a Path object, try to access it
    """
    try:
        result: pd.DataFrame = pd.read_csv(path)
        return result
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


def get_grid_params(plot_count: int, cols_count: int) -> tuple[int, int]:
    """
    Compute the number of lines and columns for
    the grid of subplots

    plot_count: the number of histograms we need (one per feature)
    cols_count: number of columns (5, as in the pdf)

    Return:
    tuple (rows_count, cols_count)
    """
    if cols_count == 0:
        print(
            "The number of column in the figure cannot be zero. Abort",
            file=sys.stderr,
        )
        sys.exit(3)
    rows_count = int(np.ceil(float(plot_count) / float(cols_count)))
    return (rows_count, cols_count)


def plot_histogram(data: pd.DataFrame, output: str) -> None:
    """
    Plot the histogram
    """
    features: list[str] = []
    cols = list(data.columns)
    if "knight" in data.columns:
        target = data["knight"]
        i = 0
        while i < len(cols):
            col = cols[i]
            if col != "knight":
                features.append(col)
            i += 1
    else:
        target = None
        features = list(data.columns)

    rows_count, cols_count = get_grid_params(len(features), 5)
    fig, axes = plt.subplots(
        rows_count,
        cols_count,
        figsize=(15, 10),
        constrained_layout=True,
    )
    if isinstance(axes, np.ndarray):
        ax_list = axes.ravel().tolist()
    else:
        ax_list = [axes]
    # The default colour is not clear, use customized colour
    label_colours: list[str] = ["orange", "skyblue"]
    i = 0
    while i < len(features):
        feature_name = features[i]
        ax = ax_list[i]
        if target is not None:
            values = data[feature_name].dropna().to_numpy(dtype=float)
            if values.size == 0:
                ax.set_title(
                    feature_name,
                )
                i += 1
                continue
            bin_edges = np.histogram_bin_edges(values, bins=120)
            labels = target.dropna().unique()
            labels = sorted(labels)
            k = 0
            while k < len(labels):
                label = labels[k]
                mask = target == label
                class_values = (
                    data[feature_name][mask].dropna().to_numpy(dtype=float)
                )
                # Assign colour based on index, here we have 2 labels
                colour: str = label_colours[k % len(label_colours)]
                ax.hist(
                    class_values,
                    bins=bin_edges,
                    alpha=0.6,
                    label=str(label),
                    color=colour,
                )
                k += 1
            ax.set_title(feature_name, fontsize=9)
            ax.tick_params(axis="both", labelsize=7)
            ax.legend(loc="upper right", fontsize=7)
        else:
            values = data[feature_name].dropna().to_numpy(dtype=float)
            ax.hist(
                values, bins=40, alpha=0.6, label="knight", color="skyblue"
            )
            ax.set_title(feature_name, fontsize=9)
            ax.legend(loc="upper right", fontsize=7)
        i += 1
    # try:
    #     fullwindow = plt.get_current_fig_manager()
    #     fullwindow.full_screen_toggle()
    # except Exception as e:
    #     print(
    #         f"Unable to display the figure on full-screen mode: {e}",
    #         file=sys.stderr,
    #     )
    fig.savefig(output, dpi=300)
    plt.show()
    plt.close()


def main() -> int:
    """
    main function
    """
    try:
        data: pd.DataFrame = load_csv(TEST_FILE)
        plot_histogram(data, "histogram1.png")
        data2: pd.DataFrame = load_csv(TRAIN_FILE)
        plot_histogram(data2, "histogram2.png")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
