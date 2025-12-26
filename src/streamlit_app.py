#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import joblib
import streamlit as st

# ---- DL ----
from dl_model import dl_predict

# ---------- Session State ----------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ---------- text cleaning ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- path helpers ----------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths():
    root = project_root()
    out = root / "outputs"
    return {
        "pipeline": out / "pipeline.joblib",
        "model": out / "model.joblib",
        "vectorizer": out / "vectorizer.joblib",
    }


def load_pipeline_or_parts(pipeline_path: Path, model_path: Path, vectorizer_path: Path):
    if pipeline_path and pipeline_path.exists():
        return joblib.load(pipeline_path), None, None
    if model_path.exists() and vectorizer_path.exists():
        clf = joblib.load(model_path)
        vec = joblib.load(model_path)
        return None, clf, vec
    return None, None, None


# ---------- streamlit app ----------
def main():
    # ---- CLI args ----
    dp = default_paths()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", default=str(dp["pipeline"]))
    ap.add_argument("--model", default=str(dp["model"]))
    ap.add_argument("--vectorizer", default=str(dp["vectorizer"]))
    args, _ = ap.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    model_path = Path(args.model).resolve()
    vectorizer_path = Path(args.vectorizer).resolve()

    # ---- UI ----
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
    st.title("📰 Fake News & Misinformation Detector")
    st.caption("Integrated ML + DL + Generative AI System")

    # ---- Sidebar ----
    with st.sidebar:
        st.subheader("ML Model Artifacts")
        st.code(
            f"pipeline:  {pipeline_path}\n"
            f"model:     {model_path}\n"
            f"vectorizer:{vectorizer_path}"
        )

    # ---- Load ML ----
    pipe, clf, vec = load_pipeline_or_parts(
        pipeline_path, model_path, vectorizer_path
    )

    if pipe is None and (clf is None or vec is None):
        st.error("ML model artifacts not found in outputs/.")
        st.stop()

    # ---- Input ----
    txt = st.text_area("Paste headline or article text:", height=200)
    threshold = st.slider("ML FAKE probability threshold", 0.05, 0.95, 0.50, 0.01)

    # ---- Analyze ----
    if st.button("Analyze"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            st.session_state.analyzed = True
            st.session_state.txt = txt

            s = clean_text(txt)

            # ---- ML ----
            if pipe is not None:
                ml_prob = float(pipe.predict_proba([s])[0, 1])
            else:
                X = vec.transform([s])
                ml_prob = float(clf.predict_proba(X)[0, 1])

            st.session_state.ml_prob = ml_prob
            st.session_state.ml_label = "FAKE" if ml_prob >= threshold else "REAL"

            # ---- DL ----
            dl_label, dl_conf = dl_predict(txt)
            st.session_state.dl_label = dl_label
            st.session_state.dl_conf = dl_conf

            # ---- Final ----
            # ---- Final Decision (Option 2: Risk-aware fusion) ----
            if st.session_state.ml_label == "FAKE":
                st.session_state.final_label = "FAKE"
            elif st.session_state.ml_label == "REAL" and dl_label == "NEGATIVE":
                st.session_state.final_label = "LIKELY FAKE"
            else:
                st.session_state.final_label = "REAL"


            st.session_state.explanation = None  # reset explanation

    # ---- Output ----
    if st.session_state.analyzed:
        st.markdown("## Final Decision")
        st.metric("Result", st.session_state.final_label)

        st.markdown("### 🔹 Machine Learning")
        st.write(f"Prediction: **{st.session_state.ml_label}**")
        st.write(f"Fake probability: **{st.session_state.ml_prob:.2f}**")

        st.markdown("### 🔹 Deep Learning (DistilBERT)")
        st.write(f"Prediction: **{st.session_state.dl_label}**")
        st.write(f"Confidence: **{st.session_state.dl_conf:.2f}**")

if __name__ == "__main__":
    main()
