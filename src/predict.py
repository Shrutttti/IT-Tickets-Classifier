import argparse
import joblib
from preprocess import clean_text
import pandas as pd


def load_model(model_path: str):
    artifact = joblib.load(model_path)
    return artifact["model"]


def predict(model, data: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(data), index=data.index, name="predicted_group")


def predict_proba(model, data: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(data)
    return pd.DataFrame(proba, index=data.index, columns=model.classes_)


def main():
    parser = argparse.ArgumentParser(description="Run inference on new IT tickets")
    parser.add_argument("--data", required=True, help="Path to input CSV of new tickets")
    parser.add_argument("--model", default="lgbm_classifier.joblib", help="Path to trained model artifact (default: lgbm_classifier.joblib)")
    parser.add_argument("--output", help="Path for predictions CSV (default: <input>_predictions.csv)")
    args = parser.parse_args()

    output_path = args.output or args.data.replace(".csv", "_predictions.csv")

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    print(f"Reading tickets from {args.data}...")
    tickets = pd.read_csv(args.data)

    predictions = predict(model, tickets)
    probabilities = predict_proba(model, tickets)

    result = tickets.copy()
    result["predicted_group"] = predictions
    result["confidence"] = probabilities.max(axis=1)

    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()