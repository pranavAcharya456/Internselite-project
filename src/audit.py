import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from src import config
from src.split import TEST_CSV, TRAIN_CSV

THRESHOLD = 0.95


def bootstrap_ci(y_true, y_pred, metric, n=1000, seed=config.RANDOM_STATE, alpha=0.05):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        vals.append(metric(y_true[i], y_pred[i]))
    return (metric(y_true, y_pred), float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))

if __name__ == "__main__":
    if not (TRAIN_CSV.exists() and TEST_CSV.exists()):
        raise SystemExit("train.csv / test.csv not found. Create them first:\n"
                         "  python -m src.data_loader && python -m src.preprocess && python -m src.split")
    train, test = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
    x_tr, x_te = train["clean_no_title"].fillna(""), test["clean_no_title"].fillna("")
    vec = TfidfVectorizer(min_df=2).fit(x_tr)
    sim = cosine_similarity(vec.transform(x_te), vec.transform(x_tr))
    best = sim.max(axis=1)
    leaky = best >= THRESHOLD
    print(f"Exact duplicates train/test: {len(set(x_tr) & set(x_te))}")
    print(f"Near-duplicates (cosine >= {THRESHOLD}): {leaky.sum()} of {len(test)} test resumes "
          f"({100 * leaky.mean():.1f}%), max similarity {best.max():.3f}")

    pipe = joblib.load(config.MODEL_PATH)
    pred = pipe.predict(x_te)
    for name, mask in [("all test resumes", np.ones(len(test), bool)), ("without near-duplicates", ~leaky)]:
        print(f"{name:26s} n={mask.sum():3d}  accuracy={accuracy_score(test.label[mask], pred[mask]):.4f}  "
              f"macro-F1={f1_score(test.label[mask], pred[mask], average='macro'):.4f}")

    f1 = lambda a, b: f1_score(a, b, average="macro")
    for name, fn in [("macro-F1", f1), ("accuracy", accuracy_score)]:
        point, lo, hi = bootstrap_ci(test["label"], pred, fn)
        print(f"{name:9s} {point:.3f}  bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
