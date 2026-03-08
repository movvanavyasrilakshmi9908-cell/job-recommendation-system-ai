import streamlit as st
import pandas as pd
import fitz
from recruiter_model import RecruiterRankingSystem


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Recruiter Dashboard",
    page_icon="🎯",
    layout="wide"
)


# ---------------- CUSTOM UI THEME ----------------

st.markdown("""
<style>

body {
    background-color:#f5f7fb;
}

.card {
    background:white;
    padding:25px;
    border-radius:12px;
    border:1px solid #e5e7eb;
    box-shadow:0 2px 6px rgba(0,0,0,0.05);
}

.stButton > button {
    background-color:#2563eb;
    color:white;
    border-radius:8px;
    padding:10px 20px;
    border:none;
    font-weight:600;
}

.stButton > button:hover {
    background-color:#1e4fd6;
}

[data-testid="stFileUploader"] {
    border:2px dashed #3b82f6;
    border-radius:10px;
    padding:15px;
}

.section-title {
    font-size:22px;
    font-weight:600;
    margin-bottom:10px;
}

.candidate-card {
    background:white;
    padding:15px;
    border-radius:10px;
    border:1px solid #e5e7eb;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- MODEL CACHE ----------------

@st.cache_resource
def load_ranking_system():
    return RecruiterRankingSystem("data/resumes.csv")

ranking_system = load_ranking_system()


# ---------------- PDF HELPER ----------------

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()


# ---------------- HEADER ----------------

# ---------------- SIDEBAR ----------------

st.sidebar.header("Ranking Filters")

top_k = st.sidebar.selectbox(
    "Top Candidates",
    [10,20,50],
    index=1
)

min_exp = st.sidebar.number_input(
    "Minimum Experience",
    min_value=0,
    value=0
)


# ---------------- JD INPUT CARD ----------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Job Description</div>', unsafe_allow_html=True)

jd_text = st.text_area(
    "Paste Job Description",
    height=250
)

uploaded_file = st.file_uploader(
    "Upload JD PDF",
    type=["pdf"]
)

if uploaded_file:
    jd_text = extract_text_from_pdf(uploaded_file)
    st.success("JD extracted successfully")


if st.button("Find Best Candidates"):

    if not jd_text.strip():

        st.error("Please enter Job Description")

    else:

        with st.spinner("Ranking candidates..."):

            results = ranking_system.rank_candidates(
                job_description=jd_text,
                top_k=top_k,
                min_experience=min_exp
            )

            st.session_state.ranked_candidates = results

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- RESULTS SECTION ----------------

if "ranked_candidates" in st.session_state:

    st.markdown("## Ranked Candidates")

    candidates = st.session_state.ranked_candidates

    df = pd.DataFrame(candidates)

    display_df = df[[
        "candidate_id",
        "final_score",
        "semantic_score",
        "skill_score",
        "experience_score"
    ]]

    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")

    for cand in candidates:

        st.markdown('<div class="candidate-card">', unsafe_allow_html=True)

        st.subheader(f"Candidate {cand['candidate_id']}")

        st.write("Experience:", cand["experience"])

        st.write("Semantic Score:", cand["semantic_score"])

        st.write("Skill Score:", cand["skill_score"])

        st.write("Experience Score:", cand["experience_score"])

        st.write("Matched Skills:", ", ".join(cand["matched_skills"]))

        st.caption(cand["resume_summary"])

        st.markdown('</div>', unsafe_allow_html=True)