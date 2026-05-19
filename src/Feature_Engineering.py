"""
feature_engineering.py
-----------------------
Loads preprocessed IT ticket data, cleans it, and engineers new features.

Usage:
    python feature_engineering.py --input_pkl <path_to_tickets_processed.pkl> --output_dir <output_path>

Outputs:
    ticketsdata.csv   - cleaned and feature-engineered dataset ready for ML pipeline
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Feature engineering for IT ticket data.")
    parser.add_argument(
        "--input_pkl",
        required=True,
        help="Path to tickets_processed.pkl (output from Data_Preprocessing.py).",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory where ticketsdata.csv will be saved (default: current dir).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data(pkl_path: str) -> pd.DataFrame:
    if not os.path.exists(pkl_path):
        print(f"[ERROR] File not found: {pkl_path}", file=sys.stderr)
        sys.exit(1)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded data with shape: {data.shape}")
    return data


# ---------------------------------------------------------------------------
# Clean data
# ---------------------------------------------------------------------------

def clean_data(data: pd.DataFrame):
    """Drop anonymised columns, handle nulls, separate open vs closed tickets."""

    # Drop anonymised columns — keep originals for ML
    ticketsdata = data.drop(columns=[
        'business_service_anon', 'service_offering_anon', 'assignment_group_anon',
        'short_description_anon', 'description_anon', 'work_notes_anon', 'comments_anon'
    ], errors='ignore')

    # Check and remove duplicates
    duplicates = ticketsdata[ticketsdata.duplicated(['ticket_number'])]
    print(f"[INFO] Duplicate records found: {len(duplicates)}")
    if len(duplicates) > 0:
        ticketsdata = ticketsdata.drop_duplicates(subset=['ticket_number'], keep='first')
        print(f"[INFO] Duplicates removed. Remaining rows: {len(ticketsdata)}")
    else:
        print("[INFO] No duplicates found — nothing to remove.")

    # Separate open tickets (no resolve_time) from closed
    open_tickets = ticketsdata[ticketsdata['resolve_time'].isnull()]
    ticketsdata  = ticketsdata[~ticketsdata['resolve_time'].isnull()]
    print(f"[INFO] Open tickets set aside : {len(open_tickets)}")
    print(f"[INFO] Closed tickets remaining: {len(ticketsdata)}")

    # Drop where short_description is null
    ticketsdata = ticketsdata.dropna(subset=['short_description'])

    # Fill null description with short_description
    ticketsdata["description"] = ticketsdata["description"].fillna(ticketsdata["short_description"])

    # Fill null business_service with assignment_group (closely related)
    ticketsdata["business_service"] = ticketsdata["business_service"].fillna(ticketsdata['assignment_group'])

    # Drop records with null service_offering
    ticketsdata = ticketsdata.dropna(subset=['service_offering'])

    # Fill null incident_type with 'Incident'
    ticketsdata['incident_type'] = ticketsdata['incident_type'].fillna('Incident')

    # Drop records with null resolved — these are cancelled tickets
    ticketsdata = ticketsdata.dropna(subset=['resolved'])

    # Fill null work_notes with comments, then fill remaining nulls with 'not available'
    ticketsdata['work_notes'] = ticketsdata['work_notes'].fillna(ticketsdata['comments'])
    ticketsdata['comments']   = ticketsdata['comments'].fillna('not available')
    ticketsdata['work_notes'] = ticketsdata['work_notes'].fillna('not available')

    print(f"[INFO] Final shape after cleaning: {ticketsdata.shape}")
    return ticketsdata


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(ticketsdata: pd.DataFrame) -> pd.DataFrame:
    """Convert resolve_time to hours, apply log transform, extract date features."""

    # Convert resolve_time from seconds to hours
    ticketsdata['resolve_time'] = ticketsdata['resolve_time'] / 3600
    print(f"[INFO] resolve_time converted to hours. Skewness: {ticketsdata['resolve_time'].skew():.4f}")

    # Log transformation to reduce skewness
    ticketsdata['resolve_time_log'] = ticketsdata['resolve_time'].apply(np.log1p)
    print(f"[INFO] resolve_time_log skewness: {ticketsdata['resolve_time_log'].skew():.4f}")

    # Date-based features from 'opened' and 'resolved' timestamps
    ticketsdata["opened_date"]    = ticketsdata["opened"].dt.day
    ticketsdata["opened_month"]   = ticketsdata["opened"].dt.month
    ticketsdata["opened_weekday"] = ticketsdata["opened"].dt.dayofweek
    ticketsdata["date_diff_days"] = (ticketsdata["resolved"] - ticketsdata["opened"]).dt.days

    print(f"[INFO] New features added: opened_date, opened_month, opened_weekday, date_diff_days, resolve_time_log")
    return ticketsdata


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load
    data = load_data(args.input_pkl)

    # 2. Clean
    ticketsdata = clean_data(data)

    # 3. Engineer features
    ticketsdata = engineer_features(ticketsdata)

    # 4. Save
    csv_path = os.path.join(args.output_dir, "ticketsdata.csv")
    ticketsdata.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()