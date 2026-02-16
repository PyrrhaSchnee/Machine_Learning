#!/usr/bin/env python3
"""
Confusion_Matrix.py

Also called error matrix, it is used to visualize the
performance of an algorithm.

It displays:
- text summary : per-class precision, recall, f1-score, total
- text : overall accuracy
- image: the confusion matrix (rows = truth, columns = prediction)
The confusion matrix image is also saved to file.

------SINGLE CLASS SCENARIO--------------------------------------
TP = true positive
TN = true negative
FP = false positive
FN = false negative

Precision answers this question :
Among those predicted as positive, how many are truly positive ?
Precision = TP / (TP + FP)

Recall answers this question :
How many positive are successfully identified as positive ?
Recall = TP / (TP + FN)

F1-score is just a metric:
Harmonic mean of precision and recall. This gets low when
the difference between precision and recall are too large
(like high recall with low precision, or low recall with high precision)

F1 = 2 * precision * recall / (precision + recall);
if precision + recall = 0, then F1 = 0

Total (also called support) answers this question :
How many real positive are there ?
Total = TP + FN

Accuracy measures the performance of the algorithm:
the higher the better

Accuracy = (TP + TN) / (TP + TN + FP + FN)

For multi-class (multi-group) matrix:
Accuracy = (sum of diagonal) / (sum of everything)

--------MULTIPLE CLASSES SCENARIO-------------------
For a given class c in an NxN confusion matrix (always rows = truth, columns = prediction):
matrix[i][j] where i = row = truth, j = column = prediction

TP(c) = matrix[c][c] = real c and predicted as c;
FP(c) = (sum of all column c) - TP(c)
FN(c) = (sum of all row c) - TP(c)
TN(c) = ALL - TP(c) - FP(c) - FN(c)
"""

from __future__ import annotations
import sys
import traceback
import matplotlib.pyplot as plt

PREDICTION_FILE: str = "predictions.txt"
TRUTH_FILE: str = "truth.txt"


def load_txt(path: str) -> list[str]:
    """
    Open the txt file and load its content then
    return a list[str]
    """
    ret: list[str] = []
    with open(path, "r", encoding="utf-8") as data:
        ret = [line.strip() for line in data.readlines()]
    if any(line == "" for line in ret):
        print(f"Empty line found in the file: {path}\nAbort", file=sys.stderr)
        sys.exit(4)
    return ret


def compute_line(
    matrix: list[list[int]], class_idx: int
) -> tuple[float, float, float, int, float, int]:
    """
    Look at the beginning of this file for formulas
    Compute the 6 indicators to be displayed :
    precision, recall, f1-score, (sub)total, accuracy, (real)total
    Return them as a tuple
    """
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp = float(matrix[class_idx][class_idx])
    fp = float(sum(matrix[r][class_idx] for r, temp in enumerate(matrix)) - tp)
    fn = float(sum(matrix[class_idx]) - tp)
    total: int = sum(sum(x) for x in matrix)
    # print(f"[DEBUG] tp: {tp} | fp: {fp} | fn: {fn}")
    tn = float(total - tp - fp - fn)
    if (tp + fp) == 0.0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0.0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return (
        precision,
        recall,
        f1,
        int(tp + fn),
        (tp + tn) / float(total),
        total,
    )


def add_space_in_line(line: str, number: int) -> str:
    """
    Added the indicated number of white space into the string and return it
    """
    i: int = 0
    while i < number:
        line += " "
        i += 1
    return line


def display_result(matrix: list[list[int]], labels: list[str]) -> None:
    """
    Display the data in text format and plot a figure
    """
    # row: list[str] = []
    class_idx: int = 0
    accuracy: float = 0
    total: int = 0
    subtotal: int = 0
    title: list[str] = ["precision", "recall", "f1-score", "total"]
    # display the "empty" left upper corner of the result
    space_nbr: int = 0
    i: int = 0
    while i < len(labels):
        if len(labels[i]) > space_nbr:
            space_nbr = len(labels[i])
        i += 1
    first_line: str = ""
    first_line = add_space_in_line(first_line, space_nbr + 3)
    i = 0
    while i < len(title):
        first_line += title[i]
        first_line += "| "
        i += 1
    # print the header
    print(first_line)
    class_idx = 0
    # print the jedi and sith and the 4 metrics
    while class_idx < len(labels):
        precision, recall, f1, subtotal, accuracy, total = compute_line(
            matrix, class_idx
        )
        print(
            f"{labels[class_idx]} |{precision:>9.2f} |{recall:>6.2f} |{f1:>8.2f} |{subtotal:>5} |"
        )
        class_idx += 1
    # print accuracy
    print(f"\naccuracy                     {accuracy} |  {total} |")
    # print matrix values
    print(
        f"\n[[{matrix[0][0]} {matrix[0][1]}]\n [{matrix[1][0]} {matrix[1][1]}]]"
    )
    # plot image
    size = len(labels)
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(matrix)
    fig.colorbar(im)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    row: int = 0
    col: int = 0
    while row < size:
        col = 0
        while col < size:
            bg_colour = im.cmap(im.norm(matrix[row][col]))
            lum = (
                0.299 * bg_colour[0]
                + 0.587 * bg_colour[1]
                + 0.114 * bg_colour[2]
            )
            if lum < 0.5:
                text_colour = "white"
            else:
                text_colour = "black"
            ax.text(
                col,
                row,
                str(matrix[row][col]),
                fontweight="bold",
                fontsize=18,
                color=text_colour,
            )
            col += 1
        row += 1
    fig.tight_layout()
    fig.savefig("cm.png", dpi=300)
    plt.show()
    plt.close(fig)


def compute_matrix(
    pred: list[str], truth: list[str], labels: list[str]
) -> list[list[int]]:
    """
    Compute the matrix and displays it

    The matrix has a type of : list[list[int]]
    """
    # here we must use set because set in python3
    # guarantees no duplicates inside it
    if len(pred) != len(truth):
        print(
            f"The prediction has a length of {len(pred)}, the truth has a length of {len(truth)}, they must have the same length.\nAbort",
            file=sys.stderr,
        )
        sys.exit(5)
    idx_by_label: dict[str, int] = {}
    row: int = 0
    col: int = 0
    matrix: list[list[int]] = []

    idx_by_label = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for x in labels] for x in labels]
    for truth_str, pred_str in zip(truth, pred):
        if truth_str not in idx_by_label:
            print(
                f"Error: Unexpected label string for truth: {truth_str}\nAbort",
                file=sys.stderr,
            )
            sys.exit(6)
        if pred_str not in idx_by_label:
            print(
                f"Error: Unexpected label string for prediction: {pred_str}\nAbort",
                file=sys.stderr,
            )
            sys.exit(7)
        row = idx_by_label[truth_str]
        col = idx_by_label[pred_str]
        matrix[row][col] += 1
    return matrix


def main() -> int:
    """
    main function
    """
    try:
        prediction_file: str = PREDICTION_FILE
        truth_file: str = TRUTH_FILE
        if len(sys.argv) == 1:
            pass
        elif len(sys.argv) == 3:
            prediction_file = sys.argv[1]
            truth_file = sys.argv[2]
        else:
            print(
                f"Usage: ./{sys.argv[0]} [path to predictions.txt] [path to truth.txt]",
                file=sys.stderr,
            )
            return 2
        prediction_data: list[str] = load_txt(prediction_file)
        truth_data: list[str] = load_txt(truth_file)
        labels = list(sorted(set(prediction_data + truth_data)))
        matrix: list[list[int]] = compute_matrix(
            prediction_data, truth_data, labels
        )
        display_result(matrix, labels)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # tb = traceback.TracebackException.from_exception(e, capture_locals=True)
        # print("".join(tb.format()), file=sys.stderr)
        # raise
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
