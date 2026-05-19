"""
process_tickets.py
------------------
Loads raw IT ticket Excel files, anonymises sensitive fields,
and saves the result as both a pickle and a CSV file.

Usage:
    python process_tickets.py --input_dir <path_to_xlsx_folder> [--output_dir <output_path>]

Outputs:
    tickets_processed.pkl
    tickets_processed.csv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import spacy


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Anonymise IT ticket data.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing raw .xlsx ticket files.",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory where processed files will be saved (default: current dir).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tickets(input_dir: str) -> pd.DataFrame:
    """Read all .xlsx files in input_dir and concatenate into one DataFrame."""
    xl_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    if not xl_files:
        print(f"[ERROR] No .xlsx files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    frames = [pd.read_excel(f) for f in xl_files]
    data = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded {len(data):,} rows from {len(xl_files)} file(s).")
    return data


# ---------------------------------------------------------------------------
# Cleaning & renaming
# ---------------------------------------------------------------------------

COLUMNS_TO_DROP = ["Stream", "Updated", "Contact Method", "Assigned to"]

COLUMN_RENAME_MAP = {
    "Number":                   "ticket_number",
    "Name":                     "service_offering",
    "Short description":        "short_description",
    "Business service":         "business_service",
    "Assignment group":         "assignment_group",
    "Work notes":               "work_notes",
    "Comments and Work notes":  "comments",
    "Incident state":           "incident_state",
    "Incident type":            "incident_type",
    "Reopen count":             "reopen_count",
    "Resolve time":             "resolve_time",
}

CATEGORICAL_COLUMNS = ["business_service", "service_offering", "assignment_group"]


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Drop, rename, and tidy up raw columns."""
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df.columns = df.columns.str.lower()
    df["ticket_number"] = df["ticket_number"].str.replace("INC", "1", regex=False)
    return df


# ---------------------------------------------------------------------------
# Categorical anonymisation
# ---------------------------------------------------------------------------

def anonymize_column(df: pd.DataFrame, column_name: str):
    """Replace real values with anonymous labels like '<column>_0', '<column>_1', …"""
    unique_values = df[column_name].unique()
    mapping = {v: f"{column_name}_{i}" for i, v in enumerate(unique_values)}
    return df[column_name].map(mapping), mapping


def anonymize_categorical_columns(df: pd.DataFrame):
    """Apply label anonymisation to each categorical column in place."""
    mappings = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[f"{col}_anon"], mappings[col] = anonymize_column(df, col)
    return df, mappings


# ---------------------------------------------------------------------------
# Text anonymisation
# ---------------------------------------------------------------------------

# Compiled regex patterns
RE_NAMED_NOTES1 = re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\s*\([A-Z][a-z]+ [A-Z][a-z]+\)')
RE_NAMED_NOTES2 = re.compile(r'\b[A-Z][a-z]+(?:\s+)?[A-Z][a-z]+\s*\([^)]*\)')
RE_FULL_NAME    = re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b')
RE_EMAIL        = re.compile(r'\S+@\S+')
RE_URL          = re.compile(r'http\S+|www\S+')
RE_IP           = re.compile(r'\b\d{1,3}(\.\d{1,3}){3}\b')
RE_HEADERS      = re.compile(r'(from:|to:|cc:|sent:).*', re.IGNORECASE)
RE_PHONES       = re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b')
RE_WHITESPACE   = re.compile(r'\s+')

NER_LABELS = {"PERSON", "ORG", "GPE", "LOC"}


def load_spacy_model(model_name: str = "en_core_web_lg"):
    """Load spaCy model; download it automatically if missing."""
    try:
        return spacy.load(model_name, disable=["parser", "tagger"])
    except OSError:
        print(f"[INFO] Downloading spaCy model '{model_name}' …")
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name, disable=["parser", "tagger"])


def anonymise_text(texts: list, nlp, batch_size: int = 128, n_process: int = 4) -> list:
    """Anonymise a list of text strings using spaCy NER + regex patterns."""
    anonymised = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        text = doc.text

        # spaCy named-entity replacement
        for ent in doc.ents:
            if ent.label_ in NER_LABELS:
                text = text.replace(ent.text, f"<{ent.label_}>")

        # Regex-based replacement
        text = RE_NAMED_NOTES1.sub("<PERSON>", text)
        text = RE_NAMED_NOTES2.sub("<PERSON>", text)
        text = RE_FULL_NAME.sub("<PERSON>", text)
        text = RE_EMAIL.sub("<EMAIL>", text)
        text = RE_URL.sub("<URL>", text)
        text = RE_IP.sub("<IP>", text)
        text = RE_HEADERS.sub("", text)
        text = RE_PHONES.sub("<PHONE>", text)
        text = RE_WHITESPACE.sub(" ", text).strip()

        anonymised.append(text)
    return anonymised


TEXT_COLUMNS = ["short_description", "description", "work_notes", "comments"]


def anonymize_text_columns(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """Apply text anonymisation to each free-text column."""
    for col in TEXT_COLUMNS:
        if col in df.columns:
            print(f"[INFO] Anonymising text column: {col} …")
            df[f"{col}_anon"] = anonymise_text(
                df[col].fillna("").tolist(),
                nlp=nlp,
                batch_size=128,
                n_process=4,
            )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load
    tickets = load_tickets(args.input_dir)
    print(tickets.info())

    # 2. Clean
    tickets = clean_tickets(tickets)

    # 3. Anonymise categorical columns
    tickets, cat_mappings = anonymize_categorical_columns(tickets)

    # 4. Load NLP model and anonymise text columns
    nlp = load_spacy_model("en_core_web_lg")
    tickets = anonymize_text_columns(tickets, nlp)

    # 5. Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    pkl_path = os.path.join(args.output_dir, "tickets_processed.pkl")
    csv_path = os.path.join(args.output_dir, "tickets_processed.csv")

    tickets.to_pickle(pkl_path)
    tickets.to_csv(csv_path, index=False)

    print(f"[INFO] Saved pickle : {pkl_path}")
    print(f"[INFO] Saved CSV    : {csv_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()