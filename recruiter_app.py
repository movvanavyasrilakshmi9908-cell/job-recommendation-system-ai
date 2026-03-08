import os
import streamlit as st
import pandas as pd
import fitz
from recruiter_model import RecruiterRankingSystem

st.set_page_config(
    page_title="Recruiter Dashboard",
    page_icon="🎯",
    layout="wide"
)

# ---------------- UI THEME ----------------

st.markdown("""
<style>

.main {
background-color:#f0f2f6;
}

.stButton>button{
background:#0078ff;
color:white;
border-radius:8px;
border:2px solid #0078ff;
font-weight:bold;
padding:8px 16px;
}

.stButton>button:hover{
background:white;
color:#0078ff;
}

.job-card{
background:white;
border-radius:12px;
padding:20px;
border:1px solid #e8e8e8;
box-shadow:0 4px 12px rgba(0,0,0,0.08);
margin-bottom:20px;
}

.tag{
display:inline-block;
padding:4px 10px;
margin:3px;
border-radius:12px;
background:#f1f5f9;
border:1px solid #e2e8f0;
font-size:13px;
}

.card{
background:white;
padding:25px;
border-radius:12px;
border:1px solid #e5e7eb;
box-shadow:0 2px 6px rgba(0,0,0,0.05);
}

.section-title{
font-size:22px;
font-weight:600;
margin-bottom:10px;
}

.score-blue{
color:#0078ff;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------

for key, default in {
    "jd_text": "",
    "jd_id": None,
    "ranked_candidates": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- MODEL ----------------

@st.cache_resource
def load_ranking_system():
    return RecruiterRankingSystem("data/resumes.csv")

ranking_system = load_ranking_system()


# ---------------- PDF HELPER ----------------

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


# ---------------- HEADER ----------------

st.title("🎯 Recruiter Candidate Ranking Dashboard")


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


# ---------------- JOB DESCRIPTION CARD ----------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">Job Description</div>', unsafe_allow_html=True)

jd_text = st.text_area("Paste Job Description", height=250)

file = st.file_uploader("Upload JD PDF", type=["pdf"])

if file:
    st.session_state.jd_text = extract_text_from_pdf(file)

elif jd_text:
    st.session_state.jd_text = jd_text


if st.session_state.jd_text:
    st.session_state.jd_id = hash(st.session_state.jd_text) % 10**8


if st.button("Find Best Candidates"):

    if not st.session_state.jd_text.strip():

        st.error("Please enter Job Description")

    else:

        with st.spinner("Ranking candidates..."):

            st.session_state.ranked_candidates = ranking_system.rank_candidates(
                st.session_state.jd_text,
                top_k,
                min_exp
            )

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- RESULTS ----------------

if st.session_state.ranked_candidates:

    st.markdown("## Ranked Candidates")

    candidates = st.session_state.ranked_candidates

    df = pd.DataFrame(candidates)

    cols = [
        "candidate_id",
        "name",
        "phone",
        "email",
        "final_score",
        "semantic_score",
        "skill_score",
        "experience_score"
    ]

    cols = [c for c in cols if c in df.columns]

    if cols:
        st.dataframe(df[cols], use_container_width=True)

    st.markdown("---")

    for i, cand in enumerate(candidates):

        st.markdown("<div class='job-card'>", unsafe_allow_html=True)

        st.subheader(f"Candidate {cand['candidate_id']} - {cand.get('name', 'Unknown')}")

        st.markdown(f"**Phone:** {cand.get('phone', 'Unknown')} | **Email:** {cand.get('email', 'Unknown')}")

        score = cand["final_score"] * 100

        # Progress bar only outside
        st.progress(score/100)

        # ---------------- SKILLS ----------------

        if cand["matched_skills"]:
            tags = "".join(
                [f"<span class='tag'>{s}</span>" for s in cand["matched_skills"]]
            )
            st.markdown(tags, unsafe_allow_html=True)

        # ---------------- DETAILS ----------------

        with st.expander("Resume Summary"):

            st.write(cand["resume_summary"])

            st.markdown("---")

            st.caption(f"Match Score: {score:.1f}%")

            st.markdown(
                f"Experience: <span class='score-blue'>{cand['experience']} yrs</span>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"Semantic Score: <span class='score-blue'>{cand['semantic_score']}</span>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"Skill Score: <span class='score-blue'>{cand['skill_score']}</span>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"Experience Score: <span class='score-blue'>{cand['experience_score']}</span>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"Final Score: <span class='score-blue'>{cand['final_score']}</span>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

else:

    st.info("Paste a Job Description and click **Find Best Candidates** to see results.")