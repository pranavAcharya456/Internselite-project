import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.svm import LinearSVC
from src import config
from src.pipeline import build_pipeline
from src.split import load_split

RS = config.RANDOM_STATE


def build_final(name, params):
    ng = tuple(params.get("tfidf__ngram_range", (1, 2)))
    if name == "Linear SVM":
        clf = CalibratedClassifierCV(
            LinearSVC(C=params["clf__C"], class_weight="balanced", random_state=RS), cv=5)
    elif name == "Logistic Regression":
        clf = LogisticRegression(C=params["clf__C"], max_iter=2000,
                                 class_weight="balanced", random_state=RS)
    else:
        clf = RandomForestClassifier(n_estimators=params["clf__n_estimators"],
                                     max_depth=params["clf__max_depth"],
                                     class_weight="balanced", random_state=RS, n_jobs=-1)
    return build_pipeline(clf, ngram_range=ng)


if __name__ == "__main__":
    with open(config.require_file(config.ROOT / "reports" / "best_params.json", "Run: python -m src.tune"), encoding="utf-8") as f:
        best = json.load(f)
    top = max(v["cv_macro_f1"] for v in best.values())
    tied = [k for k, v in best.items() if top - v["cv_macro_f1"] <= 0.01]
    winner = "Logistic Regression" if "Logistic Regression" in tied else max(best, key=lambda k: best[k]["cv_macro_f1"])
    print("CV scores:", {k: v["cv_macro_f1"] for k, v in best.items()}, "| tied within 0.01:", tied)
    print("Selected model (by CV):", winner, best[winner])

    X_train, X_test, y_train, y_test = load_split("no_title")
    pipe = build_final(winner, best[winner]["params"]).fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    classes = pipe.classes_
    auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
    metrics = {"model": winner, "params": best[winner]["params"],
               "accuracy": accuracy_score(y_test, y_pred),
               "macro_f1": f1_score(y_test, y_pred, average="macro"),
               "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
               "roc_auc_ovr_macro": auc,
               "top3_accuracy": float(np.mean([y in classes[np.argsort(p)[-3:]]
                                               for y, p in zip(y_test, proba)]))}
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    print(json.dumps(metrics, indent=2))

    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    (config.ROOT / "reports" / "classification_report.txt").write_text(report, encoding="utf-8")
    with open(config.ROOT / "reports" / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion matrix (test set) - {winner}")
    plt.xticks(rotation=60, ha="right"); plt.tight_layout()
    plt.savefig(config.FIG_DIR / "06_confusion_matrix.png", dpi=150)

    off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes))
           for j in range(len(classes)) if i != j and cm[i, j] > 0]
    print("Top confusions (count, true -> predicted):")
    for c, t, p in sorted(off, reverse=True)[:8]:
        print(f"  {c}  {t} -> {p}")

    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)
    import platform
    import sklearn
    with open(config.MODEL_PATH.with_name("model_info.json"), "w", encoding="utf-8") as f:
        json.dump({"sklearn": sklearn.__version__, "numpy": np.__version__,
                   "python": platform.python_version(), "model": winner}, f, indent=2)
    print("Saved model ->", config.MODEL_PATH)
