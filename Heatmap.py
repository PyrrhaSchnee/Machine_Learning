#!/usr/bin/env python3
"""
Heatmap.py

Displays a heapmap based on Pearson correlation coefficient
"""

from __future__ import annotations
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback

TRAIN_FILE: str = "Train_knight.csv"
TEST_FILE: str = "Test_knight.csv"


def load_csv(path: str) -> pd.DataFrame:
    """
    Take a string or a Path object, try to access it
    """
    try:
        result: pd.DataFrame = pd.read_csv(path)
        return result
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


def knight_to_nbr(col: pd.Series) -> np.typing.NDArray[np.float64]:
    """
    Convert Jedi to 1, Sith to 0;
    """
    values = col.astype(str)
    unique = sorted(values.dropna().unique().tolist())
    if len(unique) > 2:
        print(
            f'Error: Unsupported data structure: the are {len(unique)} different values for the "knight" column. Max supported: 2\nAbort',
            file=sys.stderr,
        )
        sys.exit(3)
    # print(f"[DEBUG] unique[0] = {unique[0]}, unique[1] = {unique[1]})
    return (values == unique[0]).to_numpy(dtype=float)


def plot_heatmap(data: pd.DataFrame) -> None:
    """
    Generate a heat map based on the Pearson's correlation coef
    """
    labels: list[str] = data.columns.tolist()
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    print(f"[DEBUG]: {data.to_numpy(dtype=float)}")
    im = ax.imshow(data.to_numpy(dtype="float"), aspect="auto")
    fig.colorbar(im)
    ax.set_title("Heatmap of the Pearson Correlation Coefficient")
    ax.set_xlabel("Features")
    ax.set_ylabel("Features")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    plt.savefig("Heatmap.png", dpi=300)
    plt.show()
    plt.close(fig)


def get_gcc(data: pd.DataFrame) -> None:
    """
    Compute the Pearson correlation coefficient for
    all possible 1vs1 combinations.
    """
    if "knight" in data.columns:
        knight_num = knight_to_nbr(data["knight"])
        # print(f"[DEBUG] knight_num: {knight_num}, len of knight_num: {len(knight_num)}, len of Sprint: {len(data["Sprint"])}")
        data["knight"] = knight_num
    corr = data.corr(method="pearson", numeric_only=True)
    corr = corr.select_dtypes(include=[np.number])
    corr.to_csv("heatmap_corr.csv")
    # print(
    #     f"[DEBUG]: correlation data:\n----------------------------------------\n{corr}"
    # )
    plot_heatmap(corr)


def main() -> int:
    """
    main function
    """
    try:
        train_file: str = TRAIN_FILE
        if len(sys.argv) == 1:
            pass
        elif len(sys.argv) == 2:
            train_file = sys.argv[1]
        else:
            print(
                f"Usage: ./{sys.argv[0]} <path to Train_knight.csv>",
                file=sys.stderr,
            )
            return 3
        data_train: pd.DataFrame = load_csv(train_file)
        get_gcc(data_train)
        return 0
    except Exception as e:
        print(f"Error: {e}. Exit", file=sys.stderr)
        # tb = traceback.TracebackException.from_exception(
        #     e, capture_locals=True
        # )
        # print("".join(tb.format()), file=sys.stderr)
        # raise
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
