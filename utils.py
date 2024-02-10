from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import seaborn as sns
import matplotlib.pyplot as plt

"""
    Plot
"""


def plot_confusion_matrix(conf_matrix, title, ax=None):
    """
    Plot Confusion Matrix as a heatmap.

    Parameters:
    conf_matrix (numpy.ndarray): Confusion matrix to be plotted.
    title (str): Title for the plot.
    ax (matplotlib.axes.Axes, optional): Axes object to plot the confusion matrix on.

    Returns:
    None
    """
    if ax is None:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 16},
        xticklabels=["Positive", "Negative"],
        yticklabels=["Positive", "Negative"],
        ax=ax,
    )
    ax.set_title(title)


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate and print various classification metrics.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_proba (array-like, optional): Predicted probabilities for positive class.

    Returns:
    Tuple of floats: (accuracy, precision, recall, f1)
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print Metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1
