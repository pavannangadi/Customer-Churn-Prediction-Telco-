"""Evaluation helpers for the Telco churn project."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.preprocessing import RANDOM_STATE
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


def get_transformed_feature_names(pipeline):
    """Feature names after preprocessing, with the 'num__'/'cat__' prefix removed."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


def to_original_feature(transformed_name, categorical_cols):
    """Map e.g. 'Contract_Month-to-month' back to 'Contract'."""
    for col in sorted(categorical_cols, key=len, reverse=True):
        if transformed_name.startswith(col + "_"):
            return col
    return transformed_name  # numeric columns keep their own name


def coefficient_table(pipeline):
    """Logistic Regression coefficients and odds ratios, largest magnitude first."""
    t = pd.DataFrame({
        "feature": get_transformed_feature_names(pipeline),
        "coefficient": pipeline.named_steps["model"].coef_[0],
    })
    t["odds_ratio"] = np.exp(t["coefficient"])
    t["abs_coef"] = t["coefficient"].abs()
    return t.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def impurity_importance_table(pipeline, categorical_cols):
    """Random Forest impurity-based importance per transformed (one-hot) column."""
    t = pd.DataFrame({
        "transformed_feature": get_transformed_feature_names(pipeline),
        "importance": pipeline.named_steps["model"].feature_importances_,
    })
    t["feature"] = t["transformed_feature"].map(
        lambda n: to_original_feature(n, categorical_cols))
    return t


def permutation_importance_table(pipeline, X, y, scoring="roc_auc", n_repeats=20):
    """Permutation importance on the ORIGINAL 19 columns, through the full pipeline."""
    r = permutation_importance(pipeline, X, y, scoring=scoring,
                               n_repeats=n_repeats, random_state=RANDOM_STATE)
    return (pd.DataFrame({"importance_mean": r.importances_mean,
                          "importance_std": r.importances_std}, index=X.columns)
            .sort_values("importance_mean", ascending=False))