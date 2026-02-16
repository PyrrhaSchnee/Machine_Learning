#!/usr/bin/env python3
"""
Feature_Selection.py

Feature selection via multi-collinearity detection using
VIF(Variance Inflation Factor)

For one feature xi, regress it against all other features
then compute R**2i, this answers the question: can this
feature be deducted/explained/implicitly described by other
features?
If yes, then we don't really need to keep this feature without
losing too much information.
If not, then we need to keep this feature, otherwise we lose
information to an unacceptable extend.

VIF(xi) = 1 / (1 - R**2i)
Tolerance(xi) = 1 / VIF = 1 - R**2i
R = 1 - SSres / SStot

for example, if a feature has VIF=20, then it means the information
brought by it is redundant with other features, so we can safely
ignore this feature.

Here we remove features one by one until VIF < 5;
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
INF_THRESHOLD: float = 0.999999999


def load_csv(path: str) -> pd.DataFrame:
    """
    load the csv data, return a dataframe
    """
    try:
        result: pd.DataFrame = pd.read_csv(path)
        return result
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


def remove_non_numeric(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all column that is not numeric
    """
    result: pd.DataFrame = data.copy()
    if "knights" in data.columns:
        result = result.drop(columns=["knight"])
    result = result.select_dtypes(include=[np.number]).copy()
    return result


def get_zscore(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute z scores:
    z = (x - mean) / std

    """
    result: pd.DataFrame = data.copy()
    labels: list[str] = result.columns.tolist()
    i: int = 0
    while i < len(labels):
        label: str = labels[i]
        col_std: float = float(result[label].std(ddof=0))
        col_mean: float = float(result[label].mean())
        if col_std == 0.0:
            result[label] = 0.0
        else:
            result[label] = (result[label] - col_mean) / col_std
        i += 1
    return result


def get_r2(target: np.ndarray, others: np.ndarray) -> float:
    """
    target is a vector (shape: [nbr of item]),
    others is a matrix (shape: [nbr of item, nbr of other features])

    Compute R2 of the target;

    Total variation of target SStot: sum of (xi - x_mean)**2
    Remaining error after prediction SSres: sum of (xi - x_predicted)**2
    R2: 1 - (SSres / SStot);

    Exception: if SStot == 0.0, return R2 = 1.0;

    Here, x_predicted doesn't mean predict the future because we haven't
    predicted anything at this stage. It means the model's fitted values
    on the same set of data used to fit it.

    """
    if others.size == 0:
        return 0.0
    i: int = 0
    # number of items in the target arary
    size: int = target.shape[0]
    x_mean = float(np.mean(target))
    # arr_intercept is used as the intercept
    arr_intercept: np.ndarray = np.ones((size, 1), dtype=float)
    arr_design: np.ndarray = np.concatenate([arr_intercept, others], axis=1)
    # beta is the vector of regression coefficients; linalg = linear algebra;
    # lstsq = get the least squares, least square is way to represent the error
    # rcond=None means using the default cutoff value so small values are
    # considered as zero
    beta = np.linalg.lstsq(arr_design, target, rcond=None)[0]
    arr_pred: np.ndarray = arr_design @ beta
    ss_res = float(np.sum((target - arr_pred) ** 2))
    ss_tot = float(np.sum((target - x_mean) ** 2))
    # print(f|arr_design.shape, others.shape, beta.shape)
    if ss_tot == 0.0:
        return 1.0
    return float(1 - (ss_res / ss_tot))


def get_vif(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF and Tolerance for each feature

    Return:
    Columns: ['VIF', 'Tolerance'] indexed by feature name string
    """
    labels: list[str] = data.columns.tolist()
    arr_x: np.ndarray = data.to_numpy(dtype=float)
    vifs: list[float] = []
    i: int = 0
    """
    The following loop compute VIF value for each feature by
    treating that column as the thing to be predicted from all
    other columns.

    For each feature xi, we do 3 things:
    1) build a regression to predict xi using all other features {xj, j != i};
    here the word "regression" simply means "predict a continuous number",
    like 12.3, 0.46, 156.5646; (not a discret number)
    2) compute the value of r**2i from the regression of step 1;
    3) compute the VIF value from step 2: VIF(xi) = 1 / (1 - r**2i)
    """
    while i < len(labels):
        """
        we take arr_y for each iteration, arr_y is one column of arr_x
        """
        arr_y: np.ndarray = arr_x[:, i]
        if arr_x.shape[1] == 1:
            """
            Here, if arr_x.shape[1] == 1, then there is only one single feature
            So VIF can only be 1.0 because multi-collinearity is impossible
            """
            vif_value: float = 1.0
        else:
            """
            arr_other is arr_x without column i
            """
            arr_other: np.ndarray = np.delete(arr_x, i, axis=1)
            r2: float = get_r2(arr_y, arr_other)
            if r2 > INF_THRESHOLD:
                vif_value = float("inf")
            else:
                vif_value = 1.0 / (1.0 - r2)
        vifs.append(vif_value)
        i += 1
    result: pd.DataFrame = pd.DataFrame(index=labels)
    result["VIF"] = vifs
    result["Tolerance"] = 1.0 / result["VIF"]
    return result


def clear_vif(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all those who have a value of VIF >= 5
    """
    while True:
        max_label: str = ""
        max_value: float = -999999.0
        vif: pd.DataFrame = get_vif(get_zscore(data.copy()))
        # print(f"[DEBUG]: {vif.index.tolist()}")
        i: int = 0
        should_drop: bool = False
        while i < vif.shape[0]:
            if max_value < vif["VIF"].iloc[i] and vif["VIF"].iloc[i] >= 5.0:
                max_value = vif["VIF"].iloc[i]
                max_label = vif.index.tolist()[i]
                # print(f"[DEBUG]: {max_label}\n{max_value}")
                should_drop = True
            i += 1
        if should_drop:
            data = data.drop(columns=[max_label])
        else:
            break
    return data


def main() -> int:
    """
    main function
    """
    try:
        """Load csv data and remove non-numeric columns"""
        data: pd.DataFrame = remove_non_numeric(load_csv(TRAIN_FILE))
        """Compute Z-scores"""
        data = get_zscore(data)
        data = clear_vif(data)
        vif_table = get_vif(data)
        print(vif_table)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
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
