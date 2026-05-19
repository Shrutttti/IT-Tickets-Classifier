import argparse
import joblib
from preprocess import clean_text
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

def evaluate(model, X, y, split_name: str = "Test"):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y, y_proba, multi_class="ovr")

    print(f"\n{'='*50}")
    print(f"  {split_name} Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy     : {accuracy:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Weighted F1  : {weighted_f1:.4f}")
    print(f"  ROC-AUC (OvR): {auc:.4f}")
    print()
    print(classification_report(y, y_pred, zero_division=0))

    return y_pred


def plot_confusion_matrix(y_true, y_pred, output_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix — LightGBM Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LightGBM classifier on held-out test set")
    parser.add_argument("--model", default="lgbm_classifier.joblib", help="Path to saved model artifact (default: lgbm_classifier.joblib)")
    parser.add_argument("--cache", default="preprocessed_cache.joblib", help="Path to preprocessed cache containing raw test set (default: preprocessed_cache.joblib)")
    parser.add_argument("--cm-output", default="confusion_matrix.png", help="Path to save confusion matrix image (default: confusion_matrix.png)")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)["model"]

    print(f"Loading raw test set from {args.cache}...")
    cache = joblib.load(args.cache)
    X_test_raw = cache["X_test_raw"]
    y_test     = cache["y_test"]

    y_test_pred = evaluate(model, X_test_raw, y_test, split_name="Test")
    plot_confusion_matrix(y_test, y_test_pred, output_path=args.cm_output)


if __name__ == "__main__":
    main()