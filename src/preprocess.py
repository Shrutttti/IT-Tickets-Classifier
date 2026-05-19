import argparse
import joblib
import pandas as pd
import string
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split


ORDINAL_COLUMNS = ["urgency", "priority", "impact"]
CATEGORICAL_COLUMNS = ["business_service", "service_offering", "incident_type"]
TEXT_COLUMNS = ["short_description", "description"]

DROP_COLUMNS = [
    "ticket_number", "opened", "work_notes", "resolved",
    "incident_state", "comments", "reopen_count", "assignment_group",
    "resolve_time", "resolve_time_log", "date_diff_days",
]

MIN_GROUP_SIZE = 30


def load_data(filepath: str) -> tuple:
    df = pd.read_csv(filepath)
    counts = df["assignment_group"].value_counts()
    valid_groups = counts[counts >= MIN_GROUP_SIZE].index
    df = df[df["assignment_group"].isin(valid_groups)]
    X = df.drop(columns=DROP_COLUMNS)
    y = df["assignment_group"]
    return X, y


def split_data(X, y) -> tuple:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def clean_text(a):
    a = pd.Series(a)
    return (
        a.fillna("")
         .str.lower()
         .str.translate(str.maketrans("", "", string.punctuation))
         .str.replace(r"\n+", " ", regex=True)
         .str.strip()
    )


def build_preprocessor() -> ColumnTransformer:
    ordinal_encoder = OrdinalEncoder(
        categories=[
            ["1 - High", "2 - Medium", "3 - Low"],
            ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low", "5 - Planning"],
            ["1- High", "2 - Medium", "3 - Low"],
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    onehot_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True)

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", ordinal_encoder),
    ])
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder),
    ])

    short_desc_tfidf = TfidfVectorizer(
        stop_words="english", max_features=2000, min_df=3, max_df=0.8,
        ngram_range=(1, 2), lowercase=True,
    )
    desc_tfidf = TfidfVectorizer(
        stop_words="english", max_features=5000, min_df=3, max_df=0.8,
        ngram_range=(1, 2), lowercase=True,
    )

    def make_text_pipeline(tfidf):
        return Pipeline([
            ("cleaner", FunctionTransformer(clean_text, validate=False)),
            ("tfidf", tfidf),
        ])

    text_transformer = ColumnTransformer([
        ("short", make_text_pipeline(short_desc_tfidf), "short_description"),
        ("desc", make_text_pipeline(desc_tfidf), "description"),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", ordinal_pipeline, ORDINAL_COLUMNS),
            ("nom", nominal_pipeline, CATEGORICAL_COLUMNS),
            ("text", text_transformer, TEXT_COLUMNS),
        ],
        remainder="passthrough",
    )
    return preprocessor


def main():
    parser = argparse.ArgumentParser(description="Fit preprocessor and cache transformed data for training")
    parser.add_argument("--data", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="preprocessed_cache.joblib", help="Path to save the cache (default: preprocessed_cache.joblib)")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    X, y = load_data(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Fitting preprocessor on train, transforming train and val...")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t   = preprocessor.transform(X_val)

    joblib.dump({
        "preprocessor": preprocessor,
        "X_train": X_train_t,
        "X_val": X_val_t,
        "X_test_raw": X_test,   # test data is not transformed
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }, args.output)
    print(f"Cache saved to {args.output}")


if __name__ == "__main__":
    main()