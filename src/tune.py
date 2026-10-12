import json
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from src import config
from src.pipeline import build_pipeline
from src.split import load_split

RS = config.RANDOM_STATE
NGRAMS = [(1, 1), (1, 2)]

SEARCHES = {
    "Linear SVM": (LinearSVC(class_weight="balanced", random_state=RS),
                   {"clf__C": [0.1, 0.5, 1, 5], "tfidf__ngram_range": NGRAMS}),
    "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS),
                            {"clf__C": [1, 10, 50, 100], "tfidf__ngram_range": NGRAMS}),
    "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=RS, n_jobs=-1),
                      {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 60]}),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SEARCHES)
    X_train, _, y_train, _ = load_split("no_title")
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=RS)
    out_json = config.ROOT / "reports" / "best_params.json"
    out_csv = config.ROOT / "reports" / "tuning_results.csv"
    best = {}
    if out_json.exists():
        with open(out_json, encoding="utf-8") as f:
            best = json.load(f)
    old_rows = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    for name in names:
        clf, grid = SEARCHES[name]
        gs = GridSearchCV(build_pipeline(clf), grid, cv=cv, scoring="f1_macro", n_jobs=1, refit=False)
        gs.fit(X_train, y_train)
        best[name] = {"params": {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in gs.best_params_.items()},
                      "cv_macro_f1": round(gs.best_score_, 4)}
        res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]]
        res.insert(0, "model", name)
        res["params"] = res["params"].astype(str)
        if len(old_rows):
            old_rows = old_rows[old_rows["model"] != name]
        old_rows = pd.concat([old_rows, res.round(4)])
        print(f"{name:20s} best CV macro-F1 = {gs.best_score_:.3f}  params = {gs.best_params_}", flush=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        old_rows.to_csv(out_csv, index=False)
    print("\nWinner by CV so far:", max(best, key=lambda k: best[k]["cv_macro_f1"]))
