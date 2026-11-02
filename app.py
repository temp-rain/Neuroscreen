"""NeuroScreen college-project dashboard. Run: streamlit run app.py"""
from __future__ import annotations

import html
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from feature_extractors import aggregate_child, audio_features, load_models, prototype_domain_scores, video_features

ROOT = Path(__file__).parent
st.set_page_config(page_title="NeuroScreen", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Visual design
# Palette: ink #12263A, clinic teal #0E7C7B, mist #EDF3F2, paper #FFFFFF,
#          risk low #2F8F5B / mid #C98A12 / high #C8453B
# Type:    Source Serif 4 (headlines), IBM Plex Sans (interface)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap');

:root {
    --ink: #12263A;
    --ink-soft: #4B5F73;
    --teal: #0E7C7B;
    --teal-deep: #0A5F5E;
    --mist: #EDF3F2;
    --line: #D3DFDD;
    --paper: #FFFFFF;
    --low: #2F8F5B;  --low-bg: #E4F3EA;
    --mid: #B27800;  --mid-bg: #FBF0D3;
    --high: #C8453B; --high-bg: #FBE5E2;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--ink);
}
.stApp { background: var(--mist); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 1.8rem; padding-bottom: 4rem; }

h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif; color: var(--ink); letter-spacing: -0.01em; }
h2, h3 { font-weight: 600; }
p, label, li, span { line-height: 1.55; }

/* ---------- Hero ---------- */
.ns-hero {
    background: var(--ink);
    color: #F2F7F7;
    border-radius: 18px;
    padding: 2.4rem 2.6rem;
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 2rem;
    align-items: center;
    margin-bottom: 1.2rem;
}
.ns-hero h1 {
    font-size: 3.1rem; line-height: 1.05; margin: 0 0 .7rem 0;
    color: #FFFFFF; font-weight: 700; text-align: left;
}
.ns-hero p { margin: 0; max-width: 34rem; color: #B9CBD6; font-size: 1.04rem; }
.ns-domains { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: flex-end; }
.ns-domains span {
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 999px; padding: .38rem .85rem;
    font-size: .88rem; color: #DCE9EE;
}
.ns-domains .lead { background: var(--teal); border-color: var(--teal); color: #fff; }

/* ---------- Workflow strip ---------- */
.ns-steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem; margin-bottom: 1.4rem; }
.ns-step {
    background: var(--paper); border: 1px solid var(--line); border-radius: 12px;
    padding: .8rem 1rem; display: flex; gap: .75rem; align-items: center;
}
.ns-step .n {
    width: 1.9rem; height: 1.9rem; border-radius: 50%; flex: none;
    display: grid; place-items: center; font-weight: 600; font-size: .9rem;
    background: var(--mist); color: var(--ink-soft);
}
.ns-step b { display: block; font-size: .95rem; }
.ns-step small { color: var(--ink-soft); font-size: .8rem; }
.ns-step.done { border-color: var(--teal); }
.ns-step.done .n { background: var(--teal); color: #fff; }

/* ---------- Streamlit widgets ---------- */
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--line); }
[data-testid="stFileUploader"] section {
    background: var(--paper); border: 1.5px dashed #8FB5B3; border-radius: 14px; padding: 1.4rem;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--teal); }
[data-testid="stExpander"] {
    background: var(--paper); border: 1px solid var(--line); border-radius: 12px; box-shadow: none;
}
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }

.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; padding: .6rem 1.4rem; transition: background .15s, border-color .15s;
}
.stButton > button[kind="primary"] { background: var(--teal); border: 1px solid var(--teal); color: #fff; }
.stButton > button[kind="primary"]:hover { background: var(--teal-deep); border-color: var(--teal-deep); }
.stDownloadButton > button { background: var(--paper); color: var(--teal-deep); border: 1.5px solid var(--teal); }
.stDownloadButton > button:hover { background: var(--teal); color: #fff; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 3px solid #7FD1CE; outline-offset: 2px; }

.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: .6rem 1.1rem; }
.stTabs [aria-selected="true"] { color: var(--teal-deep); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--teal); }

.stProgress > div > div > div > div { background: var(--teal); }

/* ---------- Keep text readable if Streamlit's dark theme is selected ---------- */
:root, .stApp { color-scheme: light; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp small,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stRadio"] label, .stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary p,
.stApp [data-testid="stFileUploader"] small, .stApp [data-testid="stFileUploader"] span {
    color: var(--ink);
}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stFileUploader"] small { color: var(--ink-soft); }
.stApp h1, .stApp h2, .stApp h3 { color: var(--ink); }
.ns-hero, .ns-hero h1 { color: #FFFFFF; }
.ns-hero p { color: #B9CBD6; }
.ns-domains span { color: #DCE9EE; }
.ns-domains .lead { color: #FFFFFF; }
.stApp .stButton > button[kind="primary"], .stApp .stButton > button[kind="primary"] p { color: #FFFFFF; }
.stApp .stDownloadButton > button, .stApp .stDownloadButton > button p { color: var(--teal-deep); }
.stApp .stDownloadButton > button:hover, .stApp .stDownloadButton > button:hover p { color: #FFFFFF; }
.stApp input, .stApp textarea { color: var(--ink); background: var(--paper); }
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink); }
.ns-pill.low { color: var(--low); } .ns-pill.mid { color: var(--mid); } .ns-pill.high { color: var(--high); }
.ns-note, .ns-legend { color: var(--ink-soft); }

/* ---------- Results ---------- */
.ns-child {
    display: flex; justify-content: space-between; align-items: center; gap: 1rem;
    margin: 1.6rem 0 .8rem 0; padding-bottom: .6rem; border-bottom: 1px solid var(--line);
}
.ns-child h3 { margin: 0; font-size: 1.4rem; }
.ns-child small { color: var(--ink-soft); }
.ns-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: .9rem; }
.ns-card {
    background: var(--paper); border: 1px solid var(--line); border-radius: 14px;
    padding: 1rem 1.1rem; border-top: 5px solid var(--c);
}
.ns-card.low  { --c: var(--low); }
.ns-card.mid  { --c: var(--mid); }
.ns-card.high { --c: var(--high); }
.ns-card .top { display: flex; justify-content: space-between; align-items: flex-start; gap: .6rem; }
.ns-card .name { font-weight: 600; font-size: .98rem; }
.ns-card .score { font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem; font-weight: 600; margin: .5rem 0 .55rem; }
.ns-card .score small { font-size: .85rem; color: var(--ink-soft); font-family: 'IBM Plex Sans', sans-serif; font-weight: 400; }
.ns-bar { height: 8px; border-radius: 99px; background: var(--mist); overflow: hidden; }
.ns-bar span { display: block; height: 100%; background: var(--c); border-radius: 99px; }
.ns-pill {
    border-radius: 99px; padding: .15rem .65rem; font-size: .8rem; font-weight: 600; white-space: nowrap;
}
.ns-pill.low  { background: var(--low-bg);  color: var(--low); }
.ns-pill.mid  { background: var(--mid-bg);  color: var(--mid); }
.ns-pill.high { background: var(--high-bg); color: var(--high); }
.ns-legend { display: flex; gap: 1.2rem; font-size: .85rem; color: var(--ink-soft); margin: .2rem 0 .4rem; flex-wrap: wrap; }
.ns-legend i { display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .35rem; vertical-align: -1px; }
.ns-note {
    background: var(--paper); border: 1px solid var(--line); border-left: 5px solid var(--teal);
    border-radius: 10px; padding: .9rem 1.1rem; font-size: .92rem; color: var(--ink-soft); margin-top: 1.6rem;
}

@media (max-width: 800px) {
    .ns-hero { grid-template-columns: 1fr; padding: 1.6rem; }
    .ns-hero h1 { font-size: 2.3rem; }
    .ns-domains { justify-content: flex-start; }
    .ns-steps { grid-template-columns: 1fr 1fr; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
""", unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown("""
<div class="ns-hero">
  <div>
    <h1>NeuroScreen</h1>
    <p>Early developmental risk indicators from short video, audio and a parent questionnaire, scored against DSM and ICF development domains.</p>
  </div>
  <div class="ns-domains">
    <span class="lead">Social communication</span><span>Attention and arousal</span>
    <span>Speech and language</span><span>Restricted behaviours</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_steps(has_files: bool, has_results: bool) -> None:
    steps = [
        ("1", "Upload", "Videos or audio", has_files),
        ("2", "Questionnaire", "Optional M-CHAT-R", has_files),
        ("3", "Extract", "Phases 1-4 features", has_results),
        ("4", "Review", "Risk by domain", has_results),
    ]
    cells = "".join(
        f'<div class="ns-step {"done" if done else ""}"><div class="n">{"✓" if done else n}</div>'
        f'<div><b>{title}</b><small>{sub}</small></div></div>'
        for n, title, sub, done in steps
    )
    st.markdown(f'<div class="ns-steps">{cells}</div>', unsafe_allow_html=True)


def level_for(value: float) -> str:
    return "high" if value >= .67 else "mid" if value >= .34 else "low"


LEVEL_LABEL = {"low": "Low", "mid": "Mid", "high": "High"}
LEVEL_RANK = {"low": 0, "mid": 1, "high": 2}


def render_risk_cards(record: pd.Series, risk_columns: list[str]) -> None:
    cards, worst = [], "low"
    for column in risk_columns:
        try:
            value = float(record[column])
        except (TypeError, ValueError):
            continue
        level = level_for(value)
        if LEVEL_RANK[level] > LEVEL_RANK[worst]:
            worst = level
        width = max(0.0, min(value, 1.0)) * 100
        cards.append(
            f'<div class="ns-card {level}"><div class="top"><span class="name">{html.escape(str(column))}</span>'
            f'<span class="ns-pill {level}">{LEVEL_LABEL[level]}</span></div>'
            f'<div class="score">{value:.2f}<small> / 1.00</small></div>'
            f'<div class="ns-bar"><span style="width:{width:.0f}%"></span></div></div>'
        )
    st.markdown(
        f'<div class="ns-child"><h3>{html.escape(str(record["child_id"]))}</h3>'
        f'<span class="ns-pill {worst}">Highest domain: {LEVEL_LABEL[worst]}</span></div>'
        f'<div class="ns-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("### About this tool")
    st.write("NeuroScreen is a college research prototype. It combines video, audio and questionnaire signals into a risk indicator for each development domain.")
    st.markdown("### Risk levels")
    st.markdown(
        '<div class="ns-legend" style="flex-direction:column;gap:.4rem">'
        '<span><i style="background:#2F8F5B"></i>Low: below 0.34</span>'
        '<span><i style="background:#B27800"></i>Mid: 0.34 to 0.66</span>'
        '<span><i style="background:#C8453B"></i>High: 0.67 and above</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Privacy")
    st.write("Use de-identified child IDs. Files are processed in a temporary folder and removed when extraction finishes.")

render_hero()
steps_slot = st.empty()  # filled by finish_steps() once upload state is known

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


def finish_steps(has_files: bool) -> None:
    with steps_slot.container():
        render_steps(has_files=has_files, has_results="fusion_features" in st.session_state)


# ---------------------------------------------------------------------------
# 1. Upload
# ---------------------------------------------------------------------------
st.subheader("Upload recordings")
mode = st.radio("How are these recordings grouped?", ["One child", "Multiple children"], horizontal=True)
uploads = st.file_uploader("Upload videos or audio", type=["mp4", "mov", "mkv", "avi", "mp3", "wav", "m4a"],
                           accept_multiple_files=True,
                           help="Videos run Phases 1-3 and their audio runs Phase 4. Audio files run Phase 4 only.")
if not uploads:
    finish_steps(False)
    st.info("Upload one or more files to begin. Videos run Phases 1-3 and contain audio for Phase 4; audio files run Phase 4.")
    st.stop()
finish_steps(True)

assignments = {}
if mode == "One child":
    child_id = st.text_input("Child ID (use a de-identified ID)", "child_001")
    assignments = {item.name: child_id for item in uploads}
else:
    st.caption("Assign each uploaded file to a de-identified child ID.")
    columns = st.columns(2)
    for index, item in enumerate(uploads):
        assignments[item.name] = columns[index % 2].text_input(item.name, value=item.name.split("_")[0], key=f"child_{index}")

# ---------------------------------------------------------------------------
# 2. Questionnaire
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Extract
# ---------------------------------------------------------------------------
st.subheader("Run analysis")
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
        finish_steps(True)
        st.success("Feature extraction complete.")
    except Exception as error:
        st.exception(error)
        st.info("The full Phase 4 stack is resource intensive. For a Streamlit Cloud demo, use short clips and the Whisper `tiny` model configured in feature_extractors.py.")

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
if "fusion_features" in st.session_state:
    final = st.session_state["fusion_features"]
    risks = st.session_state["prototype_risks"]
    st.subheader("Model evaluated results")
    risk_columns = [column for column in risks.columns if column != "child_id" and not column.endswith(" Level")]

    tab_summary, tab_table, tab_export = st.tabs(["Risk by domain", "Score tables", "Downloads"])

    with tab_summary:
        st.markdown(
            '<div class="ns-legend"><span><i style="background:#2F8F5B"></i>Low: below 0.34</span>'
            '<span><i style="background:#B27800"></i>Mid: 0.34 to 0.66</span>'
            '<span><i style="background:#C8453B"></i>High: 0.67 and above</span></div>',
            unsafe_allow_html=True,
        )
        for _, record in risks.iterrows():
            render_risk_cards(record, risk_columns)
        st.markdown(
            '<div class="ns-note">These scores are experimental indicators from a student prototype. '
            'They are not a diagnosis. Share any concern with a qualified clinician.</div>',
            unsafe_allow_html=True,
        )

    with tab_table:
        for _, record in risks.iterrows():
            st.markdown(f"**{record['child_id']}**")
            st.dataframe(pd.DataFrame({column: [record[column], record[f'{column} Level']] for column in risk_columns},
                                      index=["Risk score", "Risk level"]), use_container_width=True)
        with st.expander("Show extracted feature table"):
            st.dataframe(final, use_container_width=True)

    with tab_export:
        st.caption("Both files include every child in this run.")
        download_columns = st.columns(2)
        download_columns[0].download_button("Download Phase 1-5 fusion features CSV", final.to_csv(index=False).encode(), "fusion_features.csv", "text/csv", use_container_width=True)
        download_columns[1].download_button("Download prototype indicator CSV", risks.to_csv(index=False).encode(), "prototype_risk_indicators.csv", "text/csv", use_container_width=True)
