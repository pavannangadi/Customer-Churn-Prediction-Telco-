"""Evaluation helpers for the Telco churn project."""

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Compute metrics on the test set, plus train accuracy to spot overfitting."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "train_accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

def plot_confusion_matrices(models: dict, X_test, y_test, names, ncols=3):
    """Plot confusion matrices (counts) for the given model names."""
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        y_pred = models[name].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)
    for ax in axes[len(names):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def print_report(name, pipeline, X_test, y_test):
    """Print a classification report for one model."""
    y_pred = pipeline.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"], digits=3))