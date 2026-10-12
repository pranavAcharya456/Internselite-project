import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from src import config

sns.set_theme(style="whitegrid")
config.FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig_name):
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / fig_name, dpi=150)
    plt.close()
    print("saved", fig_name)


def plot_class_distribution(df):
    order = df["label"].value_counts()
    plt.figure(figsize=(9, 7))
    sns.barplot(x=order.values, y=order.index, color="steelblue")
    plt.title("Number of resumes per category")
    plt.xlabel("Count")
    plt.ylabel("")
    save("01_class_distribution.png")


def plot_text_length(df):
    words = df["text"].str.split().str.len()
    plt.figure(figsize=(8, 5))
    sns.histplot(words, bins=50, color="darkorange")
    plt.axvline(words.median(), color="k", ls="--", label=f"median = {int(words.median())}")
    plt.title("Resume length distribution (words)")
    plt.xlabel("Words per resume")
    plt.legend()
    save("02_text_length.png")
    print(f"  words: mean={words.mean():.0f}, median={words.median():.0f}, max={words.max()}")


def plot_top_words(df, categories=None, n=10):
    categories = categories or ["INFORMATION-TECHNOLOGY", "HR", "CHEF", "ACCOUNTANT"]
    vec = TfidfVectorizer(stop_words="english", max_features=5000, token_pattern=r"[a-zA-Z]{3,}")
    X = vec.fit_transform(df["text"])
    vocab = np.array(vec.get_feature_names_out())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, cat in zip(axes.ravel(), categories):
        mean_tfidf = np.asarray(X[(df["label"] == cat).values].mean(axis=0)).ravel()
        top = mean_tfidf.argsort()[::-1][:n]
        sns.barplot(x=mean_tfidf[top], y=vocab[top], ax=ax, color="seagreen")
        ax.set_title(cat)
    fig.suptitle("Top words per category (mean TF-IDF, raw text)")
    save("03_top_words.png")


def plot_title_leakage(df):
    def has_label(row):
        return row["label"].replace("-", " ").lower() in row["text"][:150].lower()
    df = df.assign(leak=df.apply(has_label, axis=1))
    pct = (df.groupby("label")["leak"].mean() * 100).sort_values()
    plt.figure(figsize=(9, 7))
    sns.barplot(x=pct.values, y=pct.index, color="crimson")
    plt.title("% of resumes whose first 150 chars contain the category name")
    plt.xlabel("%")
    plt.ylabel("")
    save("04_title_leakage.png")
    print(f"  overall leakage: {df['leak'].mean()*100:.1f}%")


if __name__ == "__main__":
    data = pd.read_csv(config.require_file(config.PROCESSED_CSV))
    print("Loaded", data.shape)
    plot_class_distribution(data)
    plot_text_length(data)
    plot_top_words(data)
    plot_title_leakage(data)
