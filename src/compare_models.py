import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from src import config
from src.pipeline import build_pipeline
from src.split import load_split

RS = config.RANDOM_STATE


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RS),
        "Naive Bayes": MultinomialNB(),
        "KNN (cosine)": KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance", algorithm="brute"),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RS, n_jobs=-1),
    }


def run(variant):
    X_train, _, y_train, _ = load_split(variant)
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=RS)
    rows = []
    for name, model in get_models().items():
        t0 = time.time()
        res = cross_validate(build_pipeline(model), X_train, y_train, cv=cv,
                             scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"})
        rows.append({"variant": variant, "model": name,
                     "macro_f1_mean": res["test_f1_macro"].mean(),
                     "macro_f1_std": res["test_f1_macro"].std(),
                     "accuracy_mean": res["test_accuracy"].mean(),
                     "seconds": time.time() - t0})
        print(f"[{variant:10s}] {name:20s} macroF1={rows[-1]['macro_f1_mean']:.3f} "
              f"± {rows[-1]['macro_f1_std']:.3f}  acc={rows[-1]['accuracy_mean']:.3f}  ({rows[-1]['seconds']:.0f}s)")
    return rows


def plot_results(results):
    models = list(results["model"].unique())
    x = np.arange(len(models)); w = 0.38
    plt.figure(figsize=(9, 5))
    for k, (variant, color) in enumerate([("with_title", "#c0392b"), ("no_title", "#2e86c1")]):
        d = results[results["variant"] == variant].set_index("model").loc[models]
        plt.bar(x + (k - 0.5) * w, d["macro_f1_mean"], w, yerr=d["macro_f1_std"], capsize=3,
                color=color, label=variant)
    plt.xticks(x, models, rotation=15); plt.ylim(0, 1)
    plt.ylabel("CV macro-F1 (5-fold, train set; bars = ±1 std)")
    plt.title("Model comparison: with vs without job-title header")
    plt.legend(title="variant"); plt.tight_layout()
    plt.savefig(config.FIG_DIR / "05_model_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    results = pd.DataFrame(run("with_title") + run("no_title"))
    out = config.ROOT / "reports" / "model_comparison.csv"
    results.round(4).to_csv(out, index=False)

    plot_results(results)
    print("\nSaved", out)
    print(results.pivot(index="model", columns="variant", values="macro_f1_mean").round(3))
