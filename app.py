import json
import sys

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx(suppress_warning=True) is None:
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())

from src import config
from src.samples import SAMPLE_JDS as JD_TEXTS, SAMPLE_RESUME
from src.predict import MIN_CHARS, extract_text, load_model, predict_resume, pretty_category, version_warning

st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")

SAMPLE_JDS = {"(none)": None, "Data Analyst": "data_analyst", "Accountant": "accountant", "Sous Chef": "chef"}


@st.cache_resource(show_spinner="First run: training the model from Resume.csv (about 15-25 minutes). Keep this page open...")
def ensure_model():
    if not config.MODEL_PATH.exists():
        from src import build
        build.main()
    return True


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


def _load_sample_jd():
    key = SAMPLE_JDS[st.session_state["jd_choice"]]
    st.session_state["jd_text"] = JD_TEXTS[key] if key else ""


def _load_sample_resume():
    st.session_state["resume_text"] = SAMPLE_RESUME


with st.sidebar:
    st.header("About")
    st.write("Classifies a resume into one of **24 job categories** (TF-IDF + Logistic Regression) "
             "and scores its match against a pasted job description.")
    try:
        with open(config.ROOT / "reports" / "test_metrics.json", encoding="utf-8") as f:
            m = json.load(f)
        st.markdown("**Model performance (held-out test set)**\n"
                    f"- Accuracy: {m['accuracy']:.2f}\n- Macro-F1: {m['macro_f1']:.2f}\n"
                    f"- Top-3 accuracy: {m['top3_accuracy']:.2f}")
    except (OSError, KeyError, ValueError):
        st.markdown("**Model performance:** see README.")
    st.markdown("**Limitations**\n"
                "- Trained on one English, US-style dataset.\n"
                "- Many categories overlap (e.g. Finance vs Accountant).\n"
                "- Scanned PDFs are not supported (no OCR). Only the first 15 PDF pages are read; uploads are capped at 5 MB.\n"
                "- The fit score is a **heuristic**, not a validated metric.\n"
                "- Resumes can contain names or other personal signals, so this is a "
                "**decision-support demo, not for automated hiring decisions**.")
    st.caption("Privacy: this app does not save uploaded resumes, but it runs on a third-party host. "
               "Avoid uploading real personal data to a public demo.")

st.title("📄 Resume Screening System")
st.caption("Classify a resume by job category and check its fit against a job description.")

try:
    ensure_model()
except FileNotFoundError:
    st.error(f"Dataset not found at {config.RAW_CSV}. Put Resume.csv there and reload this page.")
    st.stop()
except SystemExit as e:
    st.error(f"Training failed: {e}")
    st.stop()

model = get_model()
if version_warning():
    st.warning(version_warning())

left, right = st.columns(2)
with left:
    st.subheader("1. Resume")
    uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
    st.text_area("...or paste resume text", key="resume_text", height=260,
                 help="If a file is uploaded, the file is used and this text is ignored.")
    st.button("Load sample resume", on_click=_load_sample_resume)
with right:
    st.subheader("2. Job description (optional)")
    st.selectbox("Load a sample JD", list(SAMPLE_JDS), key="jd_choice", on_change=_load_sample_jd)
    st.text_area("Paste job description", key="jd_text", height=260)

go = st.button("Analyze", type="primary")

if go:
    text = ""
    try:
        if uploaded is not None:
            text = extract_text(uploaded, filename=uploaded.name)
        else:
            text = st.session_state.get("resume_text", "")
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if not text.strip():
        st.info("Please upload a resume or paste its text first.")
        st.stop()

    with st.spinner("Analyzing..."):
        res = predict_resume(text, st.session_state.get("jd_text", ""), pipe=model)

    for w in res["warnings"]:
        st.warning(w)

    if not res["top_predictions"]:
        st.error(f"No prediction could be made (extracted {res['n_chars']} characters; "
                 f"at least {MIN_CHARS} are needed).")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Predicted job category")
        for i, p in enumerate(res["top_predictions"]):
            label = f"**{pretty_category(p['category'])}**" if i == 0 else pretty_category(p["category"])
            st.markdown(f"{i + 1}. {label}: {p['probability']:.0%}")
            st.progress(min(max(p["probability"], 0.0), 1.0))
        st.caption("On the test set the true category was within the top 3 predictions far more often than "
                   "first (see sidebar), so look beyond the first choice when categories overlap.")
    with c2:
        st.subheader(f"Skills found ({len(res['skills'])})")
        st.write(", ".join(f"`{s}`" for s in res["skills"]) if res["skills"] else "No dictionary skills found.")

    fit = res["fit"]
    if fit:
        st.divider()
        st.subheader("Fit against the job description")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fit score (relative)", f"{fit['fit'] * 100:.0f} / 100")
        m2.metric("Text similarity (TF-IDF cosine)", f"{fit['cosine'] * 100:.0f}%")
        m3.metric("Skill overlap", "n/a" if fit["skill_overlap"] is None else f"{fit['skill_overlap']:.0%}")
        if fit["cosine_only"]:
            st.info("The job description mentions no skills from our dictionary, so the score uses text similarity only.")
        g, r = st.columns(2)
        with g:
            st.markdown("**✅ Matched skills**")
            st.write(", ".join(f"`{s}`" for s in fit["matched_skills"]) or "None")
        with r:
            st.markdown("**❌ Missing skills**")
            st.write(", ".join(f"`{s}`" for s in fit["missing_skills"]) or "None")
        with st.expander("How is the fit score computed?"):
            st.write("fit = 0.6 × TF-IDF cosine similarity + 0.4 × (matched skills ÷ skills required by the JD). "
                     "The weights are a heuristic choice; there is no ground truth for 'fit'. "
                     "Use the score to compare candidates against the **same** job description, "
                     "not as an absolute verdict. In our own sanity check (72 resumes x 3 job descriptions) the text-similarity part alone "
                     "picked the right job description more often (94%) than the blend (79%), so treat the skill part mainly "
                     "as an explanation aid (matched / missing skills).")
