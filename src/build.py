import subprocess
import sys

from src import config

STAGES = ["src.data_loader", "src.eda", "src.preprocess", "src.split", "src.compare_models",
          "src.tune", "src.evaluate", "src.explain"]


def main():
    config.require_file(config.RAW_CSV, f"Place the dataset CSV at {config.RAW_CSV}")
    for stage in STAGES:
        print(f"\n=== {stage} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", stage], cwd=config.ROOT)
        if r.returncode != 0 and stage != "src.explain":
            raise SystemExit(f"{stage} failed")
    print("Done. Launch the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
