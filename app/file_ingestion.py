import pandas as pd
import pandera.pandas as pa
import os
from pathlib import Path
from pandera.pandas import Column, DataFrameSchema, Check
from utils import normalize_tweet



os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

# Ensure our processed folder exists
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1. Simple Ingestion + Manual Validation
# -------------------------------
def ingest_and_validate_manual(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV, checks for required columns and correct types/values.
    """
    df = pd.read_csv(file_path)
    
    # 1a. Check required columns exist
    required_cols = {"label", "tweet"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 1b. Enforce types
    # Ensure 'label' is integer
    if not pd.api.types.is_integer_dtype(df["label"]):
        try:
            df["label"] = df["label"].astype(int)
        except Exception as e:
            raise TypeError(f"Cannot convert 'label' to int: {e}")
    # Ensure 'tweet' is string
    if not pd.api.types.is_string_dtype(df["tweet"]):
        df["tweet"] = df["tweet"].astype(str)
    
    # 1c. Validate values
    invalid_labels = set(df["label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Found invalid label values: {invalid_labels}")
    if df["tweet"].isnull().any() or (df["tweet"].str.len() == 0).any():
        raise ValueError("Some tweets are empty or null.")
    
    df["cleaned_tweet"] = df["tweet"].apply(normalize_tweet)

    # save out
    out_name = Path(file_path).stem + "_manual_processed.csv"
    df.to_csv(PROCESSED_DIR / out_name, index=False)
    print(f"→ Saved manually processed data to {PROCESSED_DIR/out_name}")

    
    return df

# -------------------------------
# 2. Schema Enforcement with Pandera
# -------------------------------
from pandera import Column, DataFrameSchema, Check

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
    # Validate (lazy=True to collect all errors before raising)
    validated_df = tweet_schema.validate(df, lazy=True)

    validated_df["tweet"] = validated_df["tweet"].apply(normalize_tweet)

    # save out
    out_name = Path(file_path).stem + "_pandera_processed.csv"
    validated_df.to_csv(PROCESSED_DIR / out_name, index=False)
    print(f"→ Saved Pandera-validated data to {PROCESSED_DIR/out_name}")

    return validated_df


test_schema = DataFrameSchema({
    "id": Column(int,   nullable=False),
    "tweet": Column(str, Check.str_length(min_value=1), nullable=False),
})

def ingest_test(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = test_schema.validate(df, lazy=True)
    df["tweet"] = df["tweet"].apply(normalize_tweet)



    # save out
    out_name = Path(file_path).stem + "_pandera_processed.csv"
    df.to_csv(PROCESSED_DIR / out_name, index=False)
    print(f"→ Saved Pandera-validated data to {PROCESSED_DIR/out_name}")
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

