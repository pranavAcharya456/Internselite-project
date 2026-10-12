import hashlib

import pandas as pd
from src import config


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_dataset(path=config.RAW_CSV, expected=config.EXPECTED_SHA256):
    same = file_sha256(path) == expected
    if not same:
        print("NOTE: this CSV differs from the one used for the numbers in the README "
              "(different Kaggle version or dataset). Your results may not match exactly.")
    return same


def load_raw(path=config.RAW_CSV, text_col=config.TEXT_COL, label_col=config.LABEL_COL):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle dataset and place it there.")
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise KeyError(f"Columns {missing} not in CSV. Found: {df.columns.tolist()}. "
                       "Fix TEXT_COL / LABEL_COL in src/config.py")
    return df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})


def inspect(df, title="DATA"):
    print(f"\n===== {title} =====")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum().to_string())
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate resume texts:", df["text"].duplicated().sum())
    print("Number of classes:", df["label"].nunique())
    print("Class counts:\n", df["label"].value_counts().to_string())
    print("Text length (chars):\n", df["text"].str.len().describe().round(0).to_string())


def clean(df, min_samples=config.MIN_SAMPLES_PER_CLASS, min_chars=100):
    n0 = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[df["text"].str.len() >= min_chars]
    df = df.drop_duplicates(subset="text", keep="first")

    counts = df["label"].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if rare:
        print(f"Dropping rare classes (<{min_samples} samples): {rare}")
        df = df[~df["label"].isin(rare)]

    print(f"Rows: {n0} -> {len(df)} after cleaning")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    check_dataset()
    inspect(raw, "RAW")
    cleaned = clean(raw)
    inspect(cleaned, "CLEANED")
    config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config.PROCESSED_CSV, index=False)
    print("\nSaved ->", config.PROCESSED_CSV)
