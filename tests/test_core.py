import io
import re

import pytest

from src import config
from src.predict import (MAX_PDF_PAGES, extract_text, load_model, predict_resume, pretty_category,
                         version_warning)
from src.preprocess import clean_text, strip_title
from src.samples import SAMPLE_JDS, SAMPLE_RESUME
from src.skills import SKILLS, extract_skills, fit_score

ROOT = config.ROOT


def make_pdf(text):
    lines = [l.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for l in text.splitlines() if l.strip()]
    stream = "BT /F1 10 Tf 40 780 Td 12 TL\n" + "\n".join(f"({l}) '" for l in lines) + "\nET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out)); out += f"{i} 0 obj\n{o}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return out.encode("latin-1")


def make_docx(text):
    import docx
    d = docx.Document()
    for l in text.splitlines():
        d.add_paragraph(l)
    buf = io.BytesIO(); d.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def pipe():
    if not config.MODEL_PATH.exists():
        pytest.skip("model not trained yet: run python -m src.build")
    return load_model()


def test_clean_text_handles_bad_input():
    assert clean_text(None) == "" and clean_text("") == "" and clean_text("   ") == ""


def test_clean_text_removes_noise_and_lemmatizes():
    out = clean_text("Contact: a@b.com https://x.com Managed 25 teams & built APIs in Python3!")
    assert "@" not in out and "http" not in out and not re.search(r"\d", out)
    assert "team" in out.split() and "python" in out.split()


def test_strip_title_removes_header_only():
    raw = "HR ADMINISTRATOR\n\nHR ADMINISTRATOR       Summary     Dedicated manager"
    assert strip_title(raw).startswith("Summary") and "ADMINISTRATOR" not in strip_title(raw)


def test_skills_found_in_technical_forms():
    found = extract_skills("Built apps with Spring Boot and ReactJS, trained ML models, Oracle SQL, Python3, C++, C#.")
    for s in ("spring boot", "react.js", "machine learning", "oracle database", "python", "c++", "c#", "sql"):
        assert s in found, s


@pytest.mark.parametrize("text", [
    "Taught in the spring semester.", "Able to react quickly.", "Added 250 ml of stock.",
    "Measured angular velocity.", "Worked with Ruby Smith at Oracle Street.", "Skilled at estimating delivery times.",
    "Used Excel to excel at work.", "R&D department."])
def test_skills_no_false_positives(text):
    assert extract_skills(text) == []


def test_skills_whole_token_matching():
    assert "java" not in extract_skills("javascript developer")
    assert "sql" not in extract_skills("mysql administrator") or "mysql" in extract_skills("mysql administrator")
    assert "sql" not in extract_skills("mysql administrator")


def test_skill_dictionary_size():
    assert len(SKILLS) >= 150


def test_extract_text_all_formats_agree(tmp_path):
    paths = {"txt": tmp_path / "r.txt", "docx": tmp_path / "r.docx", "pdf": tmp_path / "r.pdf"}
    paths["txt"].write_text(SAMPLE_RESUME, encoding="utf-8")
    paths["docx"].write_bytes(make_docx(SAMPLE_RESUME))
    paths["pdf"].write_bytes(make_pdf(SAMPLE_RESUME))
    skills = {ext: extract_skills(extract_text(p)) for ext, p in paths.items()}
    assert skills["pdf"] == skills["docx"] == skills["txt"] and len(skills["pdf"]) >= 10


def test_uploaded_file_object_can_be_read_twice():
    u = io.BytesIO(make_pdf(SAMPLE_RESUME)); u.name = "r.pdf"
    assert len(extract_text(u, filename="r.pdf")) == len(extract_text(u, filename="r.pdf")) > 100


@pytest.mark.parametrize("name", ["x.xyz", "old.doc", "noext"])
def test_unsupported_types_raise_valueerror(name):
    with pytest.raises(ValueError):
        extract_text(io.BytesIO(b"abc"), filename=name)


def test_corrupt_pdf_raises_valueerror():
    with pytest.raises(ValueError):
        extract_text(io.BytesIO(b"not a pdf"), filename="bad.pdf")


def test_latin1_text_keeps_accents():
    u = io.BytesIO("Café résumé".encode("latin-1")); u.name = "x.txt"
    assert "Café" in extract_text(u, filename="x.txt")


def test_pdf_page_cap():
    assert MAX_PDF_PAGES <= 50


def test_predict_sample_resume(pipe):
    r = predict_resume(SAMPLE_RESUME, pipe=pipe)
    assert r["top_predictions"][0]["category"] == "ACCOUNTANT" and len(r["top_predictions"]) == 3
    assert abs(sum(p["probability"] for p in r["top_predictions"])) <= 1.0 + 1e-9


@pytest.mark.parametrize("text", [None, "", "too short", "1234 5678 !!!! " * 40])
def test_predict_degenerate_input_warns_not_crashes(pipe, text):
    r = predict_resume(text, pipe=pipe)
    assert r["top_predictions"] == [] and r["warnings"]


def test_predict_non_latin_text_is_flagged(pipe):
    r = predict_resume("मैं एक लेखाकार हूँ और मेरे पास छह साल का अनुभव है। " * 12, pipe=pipe)
    assert r["top_predictions"] == [] and any("non-Latin" in w for w in r["warnings"])


def test_accented_english_is_not_flagged_as_non_latin(pipe):
    r = predict_resume(SAMPLE_RESUME.replace("a", "á"), pipe=pipe)
    assert not any("non-Latin" in w for w in r["warnings"])


def test_fit_identical_text_is_one(pipe):
    t = SAMPLE_RESUME
    assert fit_score(t, t, pipe)["fit"] == pytest.approx(1.0, abs=1e-6)


def test_fit_unrelated_jd_is_low_and_gibberish_is_zero(pipe):
    t = SAMPLE_RESUME
    chef = SAMPLE_JDS["chef"]
    acc = SAMPLE_JDS["accountant"]
    assert fit_score(t, acc, pipe)["fit"] > fit_score(t, chef, pipe)["fit"]
    g = fit_score(t, "zzzz qqqq xxxx", pipe); assert g["fit"] == 0 and g["cosine_only"]


def test_version_warning_none_when_versions_match():
    assert version_warning() is None


def test_category_display_names():
    assert pretty_category("HR") == "HR" and pretty_category("BPO") == "BPO"
    assert pretty_category("INFORMATION-TECHNOLOGY") == "Information Technology"


def test_app_smoke():
    if not config.MODEL_PATH.exists():
        pytest.skip("model not trained yet: run python -m src.build")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception
    click = lambda label: [b for b in at.button if b.label == label][0].click()
    click("Load sample resume"); at.run()
    at.selectbox[0].select("Accountant").run()
    click("Analyze"); at.run()
    assert not at.exception and len(at.metric) == 3
    shown = " ".join(m.value for m in at.markdown)
    assert "Accountant" in shown


def test_dataset_fingerprint_detects_changes(tmp_path):
    from src.data_loader import check_dataset, file_sha256
    f = tmp_path / "x.csv"; f.write_text("a,b\n1,2\n", encoding="utf-8")
    assert check_dataset(f, expected=file_sha256(f)) is True
    assert check_dataset(f, expected="0" * 64) is False


def test_bootstrap_ci_is_ordered_and_degenerate_cases_are_exact():
    import numpy as np
    from sklearn.metrics import accuracy_score
    from src.audit import bootstrap_ci
    y = np.array(["a", "b"] * 50)
    assert bootstrap_ci(y, y, accuracy_score, n=50) == (1.0, 1.0, 1.0)
    pred = y.copy(); pred[:20] = "b"
    point, lo, hi = bootstrap_ci(y, pred, accuracy_score, n=200)
    assert lo <= point <= hi and hi - lo > 0


def test_missing_file_gives_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        config.require_file(tmp_path / "nope.csv")
