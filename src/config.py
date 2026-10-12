from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = Path("C:\\Users\\pranav_23053657\\OneDrive\\Desktop\\resume-screening\\Resume.csv")
PROCESSED_CSV = ROOT / "data" / "processed" / "resumes_clean.csv"
MODEL_PATH = ROOT / "models" / "best_pipeline.joblib"
FIG_DIR = ROOT / "reports" / "figures"

TEXT_COL = "Resume_str"
LABEL_COL = "Category"

EXPECTED_SHA256 = "816a7cd985a9a41e3ce8d1e83cb7e1beb1e854a89c1cf1fd0d75494faaa0559d"

for _d in (PROCESSED_CSV.parent, MODEL_PATH.parent, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
N_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 10


HINT_DATA = "Create it first: python -m src.data_loader && python -m src.preprocess && python -m src.split"


def require_file(path, hint=HINT_DATA):
    if not Path(path).exists():
        raise FileNotFoundError(f"{Path(path).name} not found. {hint}")
    return path
