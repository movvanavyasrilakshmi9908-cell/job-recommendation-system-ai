import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import os
from recruiter_model import RecruiterRankingSystem

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Recruiter Dashboard - Candidate Ranking",
    page_icon="🎯",
    layout="wide"
)

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_ranking_system():
    return RecruiterRankingSystem("data/resumes.csv")

ranking_system = load_ranking_system()

# -------------------- HELPERS --------------------
def extract_text_from_pdf(pdf_file):
    """Extract plain text from uploaded PDF JD."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

# -------------------- UI THEME --------------------
st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; }
        .stButton>button {
            background-color: #0078ff; color: white;
            border-radius: 8px; font-weight: bold;
        }
        .candidate-card {
            background-color: #ffffff; border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 20px; margin-bottom: 20px;
            border: 1px solid #e8e8e8;
        }
        .skill-tag {
            display:inline-block;margin:2px 4px;padding:2px 8px;border-radius:12px;
            background:#e6f4ea;border:1px solid #34a853;color:#1e7e34;font-size:12px;
        }
        .other-tag {
            display:inline-block;margin:2px 4px;padding:2px 8px;border-radius:12px;
            background:#f1f3f4;border:1px solid #dadce0;color:#3c4043;font-size:12px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
st.title("🎯 Recruiter Candidate Ranking Dashboard")
st.markdown("Rank candidates based on Job Description using Semantic AI and Skill Matching.")

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("Ranking Filters")
top_k = st.sidebar.selectbox("Top-K Candidates", [10, 20, 50], index=1)
min_exp = st.sidebar.number_input("Minimum Experience (Years)", min_value=0, value=0)

# -------------------- MAIN UI --------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste JD here...", height=300)
    uploaded_file = st.file_uploader("Or upload JD PDF", type=["pdf"])

    if uploaded_file:
        jd_text = extract_text_from_pdf(uploaded_file)
        st.info("JD extracted from PDF.")

    if st.button("Generate Candidates"):
        if not jd_text.strip():
            st.error("Please provide a Job Description.")
        else:
            with st.spinner("Ranking candidates..."):
                try:
                    results = ranking_system.rank_candidates(
                        job_description=jd_text,
                        top_k=top_k,
                        min_experience=min_exp
                    )
                    st.session_state.ranked_candidates = results
                    st.success(f"Found {len(st.session_state.ranked_candidates)} candidates.")
                except Exception as e:
                    st.error(f"Error ranking candidates: {e}")

with col2:
    st.subheader("Ranked Candidates")
    if "ranked_candidates" in st.session_state and st.session_state.ranked_candidates:
        candidates = st.session_state.ranked_candidates

        # Display as a Table
        df = pd.DataFrame(candidates)
        display_df = df[["candidate_id", "final_score", "semantic_score", "skill_score", "experience_score"]]
        st.dataframe(display_df.style.highlight_max(axis=0), use_container_width=True)

        st.markdown("---")
        st.subheader("Candidate Details")
        for cand in candidates:
            with st.expander(f"👤 Candidate {cand['candidate_id']} - Score: {cand['final_score']}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Experience:** {cand['experience']} years")
                    st.write(f"**Semantic Match:** {cand['semantic_score']}")
                with c2:
                    st.write(f"**Skill Match:** {cand['skill_score']}")
                    st.write(f"**Experience Match:** {cand['experience_score']}")

                st.write("**Matched Skills:**")
                skills_html = "".join([f"<span class='skill-tag'>{s}</span>" for s in cand['matched_skills']])
                st.markdown(skills_html if skills_html else "None", unsafe_allow_html=True)

                st.write("**All Skills:**")
                all_skills_html = "".join([f"<span class='other-tag'>{s}</span>" for s in cand['all_skills']])
                st.markdown(all_skills_html, unsafe_allow_html=True)

                st.write("**Resume Summary:**")
                st.caption(cand['resume_summary'])
    else:
        st.info("Upload/Paste JD and click 'Generate Candidates' to see results.")
