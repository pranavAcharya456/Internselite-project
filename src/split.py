import pandas as pd
from sklearn.model_selection import train_test_split
from src import config

PREPROCESSED = config.ROOT / "data" / "processed" / "resumes_preprocessed.csv"
TRAIN_CSV = config.ROOT / "data" / "processed" / "train.csv"
TEST_CSV = config.ROOT / "data" / "processed" / "test.csv"

VARIANTS = {"with_title": "clean_with_title", "no_title": "clean_no_title"}


def make_split():
    df = pd.read_csv(PREPROCESSED)
    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["label"],
        random_state=config.RANDOM_STATE)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    return train, test


def load_split(variant="no_title"):
    col = VARIANTS[variant]
    train = pd.read_csv(config.require_file(TRAIN_CSV))
    test = pd.read_csv(config.require_file(TEST_CSV))
    return (train[col].fillna(""), test[col].fillna(""), train["label"], test["label"])


if __name__ == "__main__":
    train, test = make_split()
    print(f"Train: {len(train)}  Test: {len(test)}")
    cmp = pd.DataFrame({"train_%": train["label"].value_counts(normalize=True) * 100,
                        "test_%": test["label"].value_counts(normalize=True) * 100,
                        "train_n": train["label"].value_counts(),
                        "test_n": test["label"].value_counts()}).round(2)
    print(cmp.to_string())
    print("\nMax class-proportion gap (train vs test): %.2f pct-points"
          % (cmp["train_%"] - cmp["test_%"]).abs().max())
