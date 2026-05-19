import argparse
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from preprocess import clean_text


def load_cache(cache_path: str) -> tuple:
    print(f"Loading preprocessed data from {cache_path}...")
    cache = joblib.load(cache_path)
    return (
        cache["preprocessor"],
        cache["X_train"],
        cache["X_val"],
        cache["y_train"],
        cache["y_val"],
    )


def tune_lightgbm(X_train_t, y_train):
    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    param_dist = {
        "n_estimators": [200, 500, 1000],
        "max_depth": [10, 15, 20],
        "learning_rate": [0.01, 0.03, 0.05],
        "num_leaves": [20, 30, 50],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.7, 0.8, 0.9],
        "reg_alpha": [0.1, 0.3, 0.5],
        "reg_lambda": [0.1, 0.3, 0.5],
    }
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train_t, y_train)
    print(f"Best params : {search.best_params_}")
    print(f"Best CV F1  : {search.best_score_:.4f}")
    return search.best_estimator_


def main():
    parser = argparse.ArgumentParser(description="Tune LightGBM on preprocessed data")
    parser.add_argument("--cache", default="preprocessed_cache.joblib", help="Path to preprocessed cache (default: preprocessed_cache.joblib)")
    parser.add_argument("--output", default="lgbm_classifier.joblib", help="Path to save trained model (default: lgbm_classifier.joblib)")
    args = parser.parse_args()

    preprocessor, X_train_t, X_val_t, y_train, y_val = load_cache(args.cache)

    print("Tuning LightGBM...")
    best_model = tune_lightgbm(X_train_t, y_train)

    print("\nValidation performance:")
    y_val_pred = best_model.predict(X_val_t)
    from sklearn.metrics import f1_score, accuracy_score
    print(f"Accuracy : {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Macro F1 : {f1_score(y_val, y_val_pred, average='macro', zero_division=0):.4f}")

    final_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", best_model),
    ])

    joblib.dump({"model": final_pipeline}, args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()