"""NeuroScreen college-project dashboard. Run: streamlit run app.py"""
from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from feature_extractors import aggregate_child, audio_features, load_models, prototype_domain_scores, video_features

ROOT = Path(__file__).parent
st.set_page_config(page_title="NeuroScreen", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(93, 170, 222, 0.35), transparent 24rem),
            radial-gradient(circle at 92% 18%, rgba(155, 122, 210, 0.26), transparent 26rem),
            linear-gradient(135deg, #f5f9ff 0%, #eef5ff 45%, #f8f3ff 100%);
        color: #16243d;
    }
    [data-testid="stHeader"] { background: rgba(0, 0, 0, 0); }
    .block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 {
        color: #152c55;
        font-weight: 750;
        letter-spacing: -0.045em;
        text-shadow: 0 2px 12px rgba(62, 103, 161, 0.13);
    }
    h2, h3 { color: #1b3d6d; }
    [data-testid="stAlert"] {
        border: 1px solid rgba(74, 117, 176, 0.20);
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(48, 83, 132, 0.08);
    }
    [data-testid="stFileUploader"], [data-testid="stExpander"], [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(82, 119, 169, 0.18);
        border-radius: 14px;
        box-shadow: 0 7px 22px rgba(42, 79, 130, 0.07);
    }
    [data-testid="stExpander"] { padding: 0.15rem 0.55rem; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 0;
        font-weight: 600;
        box-shadow: 0 6px 14px rgba(31, 75, 130, 0.18);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2069ad, #6249a9);
    }
    .stButton > button[kind="primary"]:hover { background: linear-gradient(100deg, #185b9c, #513894); }
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.86);
        color: #25528a;
        border: 1px solid rgba(61, 104, 160, 0.3);
    }
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>NeuroScreen</h1>", unsafe_allow_html=True)
st.info("DSM / ICF Development Domains based Risk Scoring")

# Phase 5 - M-CHAT-R item wording and scoring, reproduced from the supplied
# Phase5_Questionnaire_Scoring notebook.  YES is at-risk only for Q2, Q5, Q12.
QUESTIONNAIRE_ITEMS = [
    (1, "If you point at something across the room, does your child look at it?", False, {"sc": 1.0, "ar": 0.5}),
    (2, "Have you ever wondered if your child might be deaf?", True, {"sl": 0.5, "ar": 0.5}),
    (3, "Does your child play pretend or make-believe?", False, {"sc": 1.0}),
    (4, "Does your child like climbing on things?", False, {}),
    (5, "Does your child make unusual finger movements near his or her eyes?", True, {"rb": 1.0}),
    (6, "Does your child point with one finger to ask for something or to get help?", False, {"sc": 0.5, "sl": 0.5}),
    (7, "Does your child point with one finger to show you something interesting?", False, {"sc": 1.0}),
    (8, "Is your child interested in other children?", False, {"sc": 1.0}),
    (9, "Does your child show you things by bringing them to you or holding them up for you to see — not to get help, but just to share?", False, {"sc": 1.0}),
    (10, "Does your child respond when you call his or her name?", False, {"ar": 1.0, "sc": 0.5, "sl": 0.5}),
    (11, "When you smile at your child, does he or she smile back at you?", False, {"sc": 1.0}),
    (12, "Does your child get upset by everyday noises?", True, {"rb": 1.0}),
    (13, "Does your child walk?", False, {}),
    (14, "Does your child look you in the eye when you are talking to him or her, playing with him or her, or dressing him or her?", False, {"sc": 1.0, "ar": 0.5}),
    (15, "Does your child try to copy what you do?", False, {"sc": 0.5, "sl": 0.5}),
    (16, "If you turn your head to look at something, does your child look around to see what you are looking at?", False, {"ar": 1.0, "sc": 0.5}),
    (17, "Does your child try to get you to watch him or her?", False, {"sc": 1.0}),
    (18, "Does your child understand when you tell him or her to do something?", False, {"sl": 1.0, "ar": 0.5}),
    (19, "If something new happens, does your child look at your face to see how you feel about it?", False, {"sc": 0.5, "ar": 0.5}),
    (20, "Does your child like movement activities?", False, {"sc": 0.5}),
]
QUESTIONNAIRE_AXIS_MAX = {axis: sum(weights.get(axis, 0.0) for _, _, _, weights in QUESTIONNAIRE_ITEMS)
                          for axis in ("sc", "ar", "sl", "rb")}


def questionnaire_features(child_id: str, answers: dict[int, str]) -> dict:
    """Score all M-CHAT-R items and return fusion-ready, normalized axes."""
    row = {"child_id": child_id}
    axis_scores = {axis: 0.0 for axis in QUESTIONNAIRE_AXIS_MAX}
    for number, _, reverse, weights in QUESTIONNAIRE_ITEMS:
        at_risk = int((answers[number] == "Yes") if reverse else (answers[number] == "No"))
        row[f"Q{number}"] = at_risk
        for axis, weight in weights.items():
            axis_scores[axis] += at_risk * weight
    total = sum(row[f"Q{number}"] for number in range(1, 21))
    row.update({f"q_axis_{axis}": axis_scores[axis] / maximum for axis, maximum in QUESTIONNAIRE_AXIS_MAX.items()})
    row["mchatr_total"] = total
    row["mchatr_band"] = "LOW" if total <= 2 else "MODERATE" if total <= 7 else "HIGH"
    row["mchatr_refer_override"] = int(total >= 8)
    row["questionnaire_completed"] = 1
    return row

@st.cache_resource(show_spinner="Loading Phase 1-3 models...")
def get_models():
    return load_models(ROOT)

mode = st.radio("How are these recordings grouped?", ["One child", "Multiple children"], horizontal=True)
uploads = st.file_uploader("Upload videos/audio", type=["mp4", "mov", "mkv", "avi", "mp3", "wav", "m4a"], accept_multiple_files=True)
if not uploads:
    st.info("Upload one or more files to begin. Videos run Phases 1-3 and contain audio for Phase 4; audio files run Phase 4.")
    st.stop()

assignments = {}
if mode == "One child":
    child_id = st.text_input("Child ID (use a de-identified ID)", "child_001")
    assignments = {item.name: child_id for item in uploads}
else:
    st.caption("Assign each uploaded file to a de-identified child ID.")
    columns = st.columns(2)
    for index, item in enumerate(uploads):
        assignments[item.name] = columns[index % 2].text_input(item.name, value=item.name.split("_")[0], key=f"child_{index}")

st.subheader("Questionnaire")
use_questionnaire = st.checkbox("Include the optional 20-question M-CHAT-R questionnaire", value=False)
questionnaire_rows = []
if use_questionnaire:
    st.caption("Answer Yes or No for each child. The questionnaire is given greater weight than the experimental media signals.")
    for child in sorted(set(assignments.values())):
        with st.expander(f"Questionnaire - {child}", expanded=len(set(assignments.values())) == 1):
            answers = {}
            for number, prompt, _, _ in QUESTIONNAIRE_ITEMS:
                answers[number] = st.radio(f"Q{number}. {prompt}", ("Yes", "No"), key=f"{child}_q{number}", horizontal=True)
            questionnaire_rows.append(questionnaire_features(child, answers))
else:
    questionnaire_rows = [{"child_id": child, **{f"Q{number}": 0 for number in range(1, 21)},
                           **{f"q_axis_{axis}": 0.0 for axis in QUESTIONNAIRE_AXIS_MAX},
                           "mchatr_total": 0, "mchatr_band": "NOT_COMPLETED", "mchatr_refer_override": 0,
                           "questionnaire_completed": 0}
                          for child in sorted(set(assignments.values()))]

if st.button("Extract", type="primary"):
    try:
        models = get_models(); file_rows = defaultdict(list)
        progress = st.progress(0, text="Preparing files")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for index, item in enumerate(uploads):
                path = directory / item.name; path.write_bytes(item.getvalue())
                suffix = path.suffix.lower(); row = {"file": item.name}
                if suffix in {".mp4", ".mov", ".mkv", ".avi"}:
                    row.update(video_features(path, models))
                    row.update(audio_features(path))
                else:
                    row.update(audio_features(path))
                file_rows[assignments[item.name]].append(row)
                progress.progress((index + 1) / len(uploads), text=f"Processed {item.name}")
        features = pd.DataFrame([aggregate_child(rows, child) for child, rows in file_rows.items()])
        questionnaire = pd.DataFrame(questionnaire_rows)
        final = features.merge(questionnaire, on="child_id", how="left")
        risks = pd.DataFrame([{**{"child_id": record["child_id"]}, **prototype_domain_scores(record, record)} for record in final.to_dict("records")])
        for column in [name for name in risks.columns if name != "child_id"]:
            risks[f"{column} Level"] = risks[column].map(lambda value: "High" if value >= .67 else "Mid" if value >= .34 else "Low")
        st.session_state["fusion_features"] = final
        st.session_state["prototype_risks"] = risks
        st.success("Feature extraction complete.")
    except Exception as error:
        st.exception(error)
        st.info("The full Phase 4 stack is resource intensive. For a Streamlit Cloud demo, use short clips and the Whisper `tiny` model configured in feature_extractors.py.")

if "fusion_features" in st.session_state:
    final = st.session_state["fusion_features"]
    risks = st.session_state["prototype_risks"]
    st.subheader("Model evaluated Results")
    risk_columns = [column for column in risks.columns if column != "child_id" and not column.endswith(" Level")]
    for _, record in risks.iterrows():
        st.markdown(f"**{record['child_id']}**")
        st.dataframe(pd.DataFrame({column: [record[column], record[f'{column} Level']] for column in risk_columns},
                                  index=["Risk score", "Risk level"]), use_container_width=True)
    download_columns = st.columns(2)
    download_columns[0].download_button("Download Phase 1-5 fusion features CSV", final.to_csv(index=False).encode(), "fusion_features.csv", "text/csv")
    download_columns[1].download_button("Download prototype indicator CSV", risks.to_csv(index=False).encode(), "prototype_risk_indicators.csv", "text/csv")
    with st.expander("Show extracted feature table"):
        st.dataframe(final, use_container_width=True)
