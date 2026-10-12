import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src import config

sns.set_theme(style="whitegrid")


def top_features_per_class(pipe, n=15):
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    rows = []
    for k, cls in enumerate(clf.classes_):
        top = np.argsort(clf.coef_[k])[::-1][:n]
        rows += [{"class": cls, "rank": r + 1, "feature": vocab[j], "weight": clf.coef_[k, j]}
                 for r, j in enumerate(top)]
    return pd.DataFrame(rows)


def plot_top_features(df, classes, n=10, fname="07_top_features.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cls in zip(axes.ravel(), classes):
        d = df[(df["class"] == cls) & (df["rank"] <= n)]
        sns.barplot(data=d, x="weight", y="feature", ax=ax, color="teal")
        ax.set_title(cls); ax.set_xlabel("LR coefficient"); ax.set_ylabel("")
    fig.suptitle("Top words pushing a resume toward each category (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / fname, dpi=150)
    plt.close()


def lime_example(pipe, text, true_label, n_features=10):
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=list(pipe.classes_), random_state=config.RANDOM_STATE)
    probs = pipe.predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    exp = explainer.explain_instance(text, pipe.predict_proba, num_features=n_features,
                                     labels=[pred_idx], num_samples=1000)
    print(f"True: {true_label} | Predicted: {pipe.classes_[pred_idx]} (p={probs[pred_idx]:.2f})")
    pairs = exp.as_list(label=pred_idx)
    for w, v in pairs:
        print(f"   {w:20s} {v:+.3f}")
    plt.figure(figsize=(7, 4))
    words, vals = zip(*pairs[::-1])
    plt.barh(words, vals, color=["seagreen" if v > 0 else "indianred" for v in vals])
    plt.title(f"LIME: why '{pipe.classes_[pred_idx]}' (p={probs[pred_idx]:.2f})")
    plt.xlabel("Contribution to predicted class")
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "08_lime_example.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    pipe = joblib.load(config.MODEL_PATH)
    feats = top_features_per_class(pipe, n=15)
    out = config.ROOT / "reports" / "top_features_per_class.csv"
    feats.round(3).to_csv(out, index=False)
    for cls in ["INFORMATION-TECHNOLOGY", "HR", "FINANCE", "ACCOUNTANT", "CHEF", "TEACHER"]:
        print(f"{cls:24s}", ", ".join(feats[(feats['class'] == cls) & (feats['rank'] <= 8)]["feature"]))
    plot_top_features(feats, ["INFORMATION-TECHNOLOGY", "HR", "FINANCE", "ACCOUNTANT", "CHEF", "TEACHER"])

    test = pd.read_csv(config.require_file(config.ROOT / "data" / "processed" / "test.csv"))
    test["clean_no_title"] = test["clean_no_title"].fillna("")
    preds = pipe.predict(test["clean_no_title"])
    ok = np.where((preds == test["label"]) & (test["label"] == "INFORMATION-TECHNOLOGY"))[0]
    conf = pipe.predict_proba(test["clean_no_title"])[ok].max(axis=1)
    idx = int(ok[np.argmax(conf)])
    print()
    try:
        lime_example(pipe, test.loc[idx, "clean_no_title"], test.loc[idx, "label"])
    except ImportError:
        print("lime not installed -> skipping local explanation (pip install lime)")
