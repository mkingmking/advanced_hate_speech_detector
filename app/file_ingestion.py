import pandas as pd
import os
from pathlib import Path
from pandera import Column, DataFrameSchema, Check
from utils import normalize_tweet



os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

# Ensure our processed folder exists
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _check_required_columns(df: pd.DataFrame, required_cols: set) -> None:
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def _enforce_types(df: pd.DataFrame) -> None:
    if not pd.api.types.is_integer_dtype(df["label"]):
        df["label"] = df["label"].astype(int)
    if not pd.api.types.is_string_dtype(df["tweet"]):
        df["tweet"] = df["tweet"].astype(str)


def _validate_values(df: pd.DataFrame) -> None:
    invalid_labels = set(df["label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Found invalid label values: {invalid_labels}")
    if df["tweet"].isnull().any() or (df["tweet"].str.len() == 0).any():
        raise ValueError("Some tweets are empty or null.")


def _normalize_column(df: pd.DataFrame, column: str = "tweet") -> None:
    df[column] = df[column].apply(normalize_tweet)


def _save(df: pd.DataFrame, src_path: str, suffix: str) -> None:
    out_name = Path(src_path).stem + suffix
    df.to_csv(PROCESSED_DIR / out_name, index=False)
    print(f"→ Saved processed data to {PROCESSED_DIR/out_name}")

# -------------------------------
# 1. Simple Ingestion + Manual Validation
# -------------------------------
def ingest_and_validate_manual(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV, checks for required columns and correct types/values.
    """
    df = pd.read_csv(file_path)
    _check_required_columns(df, {"label", "tweet"})
    _enforce_types(df)
    _validate_values(df)
    _normalize_column(df, "tweet")
    out_name = "_manual_processed.csv"
    _save(df, file_path, out_name)
    return df

# -------------------------------
# 2. Schema Enforcement with Pandera
# -------------------------------
tweet_schema = DataFrameSchema({
    "label": Column(
        int,
        Check.isin([0, 1]),
        nullable=False,
        description="0 = not hate, 1 = hate"
    ),
    "tweet": Column(
        str,
        Check.str_length(min_value=1),   # ensures every tweet has at least 1 character
        nullable=False,
        description="Raw tweet text"
    ),
})

def ingest_and_validate_pandera(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV and validates against a Pandera schema.
    Will raise a SchemaErrors exception listing all violations.
    """
    df = pd.read_csv(file_path)
    validated_df = tweet_schema.validate(df, lazy=True)
    _normalize_column(validated_df, "tweet")
    out_name = "_pandera_processed.csv"
    _save(validated_df, file_path, out_name)
    return validated_df


test_schema = DataFrameSchema({
    "id": Column(int,   nullable=False),
    "tweet": Column(str, Check.str_length(min_value=1), nullable=False),
})

def ingest_test(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = test_schema.validate(df, lazy=True)
    _normalize_column(df, "tweet")
    out_name = "_pandera_processed.csv"
    _save(df, file_path, out_name)
    return df

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    path_train = "data/raw/train_raw.csv"
    path_test =  "data/raw/test_raw.csv"
    
    # Manual validation
    #df_manual = ingest_and_validate_manual(path)
    #print(f"Manual ingestion passed: {len(df_manual)} records")
    
    # Pandera validation
    df_train = ingest_and_validate_pandera(path_train)
    df_test = ingest_test(path_test)


    print(f"Pandera validation of training data passed: {len(df_train)} records")
    print(f"Pandera validation of test data passed: {len(df_test)} records")






    #just to test

    #samples = df_pandera["tweet"].head(20).tolist()
    #cleaned  = df_pandera["cleaned_tweet"].head(20).tolist()
    #list(zip(samples, cleaned))
    #print(samples)
    #print(cleaned)

