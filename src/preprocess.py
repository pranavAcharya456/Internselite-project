import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def _ensure_nltk():
    for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk()

EXTRA_STOPWORDS = {
    "city", "state", "company", "name", "country", "present", "current", "date",
    "year", "month", "day", "work", "worked", "working", "experience", "responsible",
    "summary", "skill", "skills", "highlight", "highlights", "professional",
    "including", "include", "included", "well", "also", "using", "used", "use",
}
STOP_WORDS = set(stopwords.words("english")) | EXTRA_STOPWORDS
_lemmatizer = WordNetLemmatizer()

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\S+@\S+")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")
_TITLE_GAP = re.compile(r" {3,}")


def strip_title(text: str) -> str:
    m = _TITLE_GAP.search(text[:200])
    if m:
        return text[m.end():]
    parts = text.split("\n", 1)
    return parts[1] if len(parts) > 1 else text


def clean_text(text, remove_title: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    if remove_title:
        text = strip_title(text)
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _NON_LETTER.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    tokens = [
        _lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(tokens)


def make_clean_column(df, remove_title: bool = False, text_col="text"):
    return df[text_col].apply(lambda t: clean_text(t, remove_title=remove_title))


if __name__ == "__main__":
    import pandas as pd
    from src import config

    df = pd.read_csv(config.PROCESSED_CSV)
    sample = df.loc[0, "text"]
    print("RAW   :", repr(sample[:150]))
    print("TITLE-STRIPPED:", repr(strip_title(sample)[:100]))
    print("CLEAN :", clean_text(sample, remove_title=True)[:150])

    df["clean_with_title"] = make_clean_column(df, remove_title=False)
    df["clean_no_title"] = make_clean_column(df, remove_title=True)
    df = df[(df["clean_no_title"].str.len() > 0)].reset_index(drop=True)
    out = config.ROOT / "data" / "processed" / "resumes_preprocessed.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")
    print("Avg words: raw=%.0f  with_title=%.0f  no_title=%.0f" % (
        df["text"].str.split().str.len().mean(),
        df["clean_with_title"].str.split().str.len().mean(),
        df["clean_no_title"].str.split().str.len().mean()))

    still = sum(r.label.replace("-", " ").lower() in strip_title(r.text)[:150].lower()
                for r in df.itertuples())
    print(f"Category name in first 150 chars AFTER stripping title: {still}/{len(df)} "
          f"({100*still/len(df):.1f}%)  (was 66.3%)")
