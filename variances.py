#!/usr/bin/env python3
"""
variances.py

Here we need to do Principal Component Analysis (PCA).
We have 31 columns (a bit too many) as input, with PCA we try to
express with fewer new columns while keeping most of the information (variation)
carried by the input, and discarding redundancy.

We consider that skills are not independent among them. So if one skill
tend to increase (or decrease) along with another skill, it's like
saying the same thing twice, so we may safely discard the information among
one of them.

PCA creates new axes called principal components:
Component1: the direction (a mix of skills) where the data varies the most
Component2: the second direction with the most variances while being orthogonal
to componnet1 so we aoivd over-lapping
Each component(skill) is a weighted sum of the original skills, like:
C1 = 0.12 * Sensitibity +0.15 * Power + ...;

Why PCA is used:
1) In original input we have 31 columns which is too many. With PCA we may only
need 5 columns while preserving at least 90% of the varaince of the original
data
2) Reduction of dedundancy, skills correlating among themselves are combined
into a same componnet.
3) Visualization of 2D plot too see the clusters(groups);
4) Reduction of noise which often has a small variance.
"""

from __future__ import annotations
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.typing import NDArray as nda
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


def knight_to_nbr(col: pd.Series) -> nda[np.float64]:
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


def get_expvar(data: pd.DataFrame) -> tuple[nda[np.float64], nda[np.float64]]:
    """
    Compute explained variance ratios in percentage for PCA components;

    1) Create the covariance matrix corresponding to the input data:
        input data is in this format: data[nbr_entry][nbr_feature];
        covariance is in this format: data[nbr_feature][nbr_feature]
    2) Compute its eigenvalues, each eigenvalue is the variance of one principle component;
    3) Normalize to get percentages
    """
    cov: pd.DataFrame = data.cov(ddof=0)
    print(f"[DEBUG] covariance\n{cov}")
    eigvals = np.linalg.eigvalsh(cov.to_numpy(dtype=float))
    eigvals = np.flip(np.sort(eigvals))
    eigvals[eigvals < 0.0] = 0.0
    eig_total = float(np.sum(eigvals))
    if eig_total == 0.0:
        print(
            "Error: the sum of the eigenvalues are 0\nAbort", file=sys.stderr
        )
        sys.exit(5)
    var_pct: nda[np.float64] = eigvals / eig_total * 100.0
    var_cumul: nda[np.float64] = np.cumsum(var_pct)
    return (var_pct, var_cumul)


def plot_var(
    var_pct: nda[np.float64], var_cumul: nda[np.float64], i: int
) -> None:
    """
    Plot the variences
    """
    print(f"[DEBUG]: var_pct:\n{var_pct}\n[DEBUG]: var_cumul\n{var_cumul}")
    x: nda[np.float64] = np.arange(1, len(var_pct) + 1, dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.plot(x, var_cumul, marker="o")
    ax.axhline(var_cumul[i - 1], color="purple")
    ax.axvline(i, color="purple")
    ax.set_title("Principal Component Analysis\nNumber of Features to retain at least 90% of information")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Percentage of Information cumulatively retained (%)")
    ax.set_xticks(x)
    ax.text(
        i + 0.2,
        88,
        f"{i} components = {var_cumul[i-1]:.2f}%",
        fontsize=14,
        color="purple",
    )
    fig.tight_layout()
    plt.savefig("variances.png")
    plt.show()
    plt.close(fig)


def get_variance(data: pd.DataFrame) -> None:
    """
    1) Replace string of the column knight by 1 and 0;
    2) Standardization of all columns
    3) Compute explained variance ratios in percentage for PCA components
    4) Display an image of this process;
    """
    # 1) knight to number
    labels = data.columns.tolist()
    if "knight" in data.columns:
        data["knight"] = knight_to_nbr(data["knight"])
    # 2) Standardization of all columns
    data = data.select_dtypes(include=[np.number]).copy()
    i: int = 0
    while i < len(labels):
        if (data[labels[i]].std(ddof=0)) == 0:
            data[labels[i]] = 1
        else:
            data[labels[i]] = (
                data[labels[i]] - float(data[labels[i]].mean())
            ) / float(data[labels[i]].std(ddof=0))
        i += 1
    # 3) Compute explained variance
    var_pct, var_cumul = get_expvar(data)
    i = 1
    while i <= len(var_cumul) and var_cumul[i - 1] < 90.00:
        i += 1
    if i > len(var_cumul):
        i = len(var_cumul)
    print(
        f"Number of components required to reach 90% of total variances: {i}"
    )
    plot_var(var_pct, var_cumul, i)


def main() -> int:
    """
    main function
    """
    try:
        train: str = TRAIN_FILE
        if len(sys.argv) == 1:
            pass
        elif len(sys.argv) == 2:
            train = sys.argv[1]
        else:
            print(
                f"Usage: ./{sys.argv[0]} [path to Train_knight.csv]",
                file=sys.stderr,
            )
            return 3
        data: pd.DataFrame = load_csv(train)
        get_variance(data)
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
