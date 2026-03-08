import os
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

st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; }
        h1, h2, h3, h4, h5, h6 { color: #262730; }
        .stButton>button {
            background-color: #0078ff; color: white;
            border: 2px solid #0078ff;
            border-radius: 8px; font-weight: bold; padding: 8px 16px;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #fff;
            color: #0078ff;
        }
        .job-card {
            background-color: #ffffff; border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 20px; margin-bottom: 25px;
            border: 1px solid #e8e8e8;
        }
        .comparison-card {
            background-color: #fff; border-radius: 12px; padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .old-rec { background-color: #fff3e0; padding: 10px; border-radius: 8px; }
        .enhanced-rec { background-color: #e3f2fd; padding: 10px; border-radius: 8px; }
        .metric-pill {
            display:inline-block;padding:6px 12px;border-radius:16px;
            background:#eef4ff;border:1px solid #cfe0ff;margin-right:8px;
            font-size:14px; font-weight: 500;
        }
        .tag {
            display:inline-block;margin:2px 6px 0 0;padding:4px 10px;border-radius:12px;
            background:#f1f5f9;border:1px solid #e2e8f0;font-size:13px;
        }
        .upload-box {
            border: 2px dashed #0078ff;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            background-color: #f8f9fa;
        }
        .card {
            background:white;
            padding:25px;
            border-radius:12px;
            border:1px solid #e5e7eb;
            box-shadow:0 2px 6px rgba(0,0,0,0.05);
        }
        .section-title {
            font-size:22px;
            font-weight:600;
            margin-bottom:10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- CONSTANTS & STATE ----------------

RATINGS_DIR = os.path.join(os.getcwd(), "data")
RATINGS_PATH = os.path.join(RATINGS_DIR, "recruiter_ratings.csv")

for key, default in {
    "jd_text": "",
    "jd_id": None,
    "ranked_candidates": [],
    "enhanced_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- MODEL CACHE ----------------

@st.cache_resource
def load_ranking_system():
    return RecruiterRankingSystem("data/resumes.csv")

ranking_system = load_ranking_system()

def ensure_ratings_dir():
    os.makedirs(RATINGS_DIR, exist_ok=True)

def save_rating(jd_id, candidate_id, rating):
    ensure_ratings_dir()
    rating_value = int(rating)

    if os.path.exists(RATINGS_PATH):
        df = pd.read_csv(RATINGS_PATH)
    else:
        df = pd.DataFrame(columns=["jd_id", "candidate_id", "rating"])

    if not df.empty:
        df["jd_id"] = df["jd_id"].astype(str)
        df["candidate_id"] = df["candidate_id"].astype(str)

    new_entry = {"jd_id": str(jd_id), "candidate_id": str(candidate_id), "rating": rating_value}

    mask = (df["jd_id"] == str(jd_id)) & (df["candidate_id"] == str(candidate_id))
    if mask.any():
        df.loc[mask, "rating"] = rating_value
        st.toast(f"🔄 Updated rating for Candidate {candidate_id}")
    else:
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        st.toast(f"💾 Added new rating for Candidate {candidate_id}")

    df["rating"] = df["rating"].astype(int)
    df.to_csv(RATINGS_PATH, index=False)

def run_rank(top_k, min_exp, use_feedback=False):
    with st.spinner("Ranking candidates..."):
        results = ranking_system.rank_candidates(
            job_description=st.session_state.jd_text,
            top_k=top_k,
            min_experience=min_exp,
            use_feedback=use_feedback,
            jd_id=st.session_state.jd_id
        )
        st.session_state.ranked_candidates = results
        st.session_state.enhanced_results = None
        st.success(f"✅ Found top {len(results)} candidates!")


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
    st.session_state.jd_text = extract_text_from_pdf(uploaded_file)
    st.success("JD extracted successfully")
elif jd_text.strip():
    st.session_state.jd_text = jd_text

if st.session_state.jd_text:
    st.session_state.jd_id = hash(st.session_state.jd_text) % (10**8)


btn_cols = st.columns(2)
with btn_cols[0]:
    if st.button("Find Best Candidates"):
        if not st.session_state.jd_text.strip():
            st.error("Please enter Job Description")
        else:
            run_rank(top_k, min_exp, use_feedback=False)

with btn_cols[1]:
    if st.button("Enhance with AI Feedback"):
        if not st.session_state.jd_text.strip():
            st.error("Please enter Job Description")
        else:
            with st.spinner("🧠 Retraining model using recruiter feedback..."):
                st.session_state.enhanced_results = ranking_system.retrain_with_feedback(
                    st.session_state.jd_text,
                    st.session_state.jd_id,
                    top_k=top_k,
                    min_experience=min_exp
                )
                if st.session_state.enhanced_results is None:
                    st.error("No feedback found to enhance with. Please provide ratings.")
                else:
                    st.session_state.ranked_candidates = st.session_state.enhanced_results["new_candidates"]

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

    for i, cand in enumerate(candidates, start=1):
        cand_id = cand['candidate_id']
        with st.container():
            st.markdown("<div class='job-card'>", unsafe_allow_html=True)

            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"#### Candidate {cand_id}")
            with cols[1]:
                st.markdown(f"<span style='float: right;'>Exp: {cand['experience']} yrs</span>", unsafe_allow_html=True)

            score_to_show = float(cand.get("adjusted_score", cand.get("final_score", 0)))
            st.slider("Match Score", 0.0, 1.0, score_to_show, disabled=True, key=f"score_{i}")

            with st.expander("Details"):
                st.write(f"**Semantic Score:** {cand['semantic_score']}")
                st.write(f"**Skill Score:** {cand['skill_score']}")
                st.write(f"**Experience Score:** {cand['experience_score']}")

                if cand.get("matched_skills"):
                    chips = "".join([f"<span class='tag'>{s.strip()}</span>" for s in cand["matched_skills"]][:8])
                    if chips:
                        st.markdown(f"**Matched Skills:** {chips}", unsafe_allow_html=True)

                st.write(f"**Summary:** {cand['resume_summary']}")

            feedback_cols = st.columns([3, 2])
            with feedback_cols[0]:
                rating = st.slider("Your Rating", 1, 5, 3, key=f"slider_{i}")
            with feedback_cols[1]:
                if st.button("Submit Feedback", key=f"rate_btn_{i}"):
                    save_rating(st.session_state.jd_id, cand_id, rating)
                    st.toast(f"⭐ You rated Candidate {cand_id} as {rating}/5")

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ENHANCED MODEL DASHBOARD ----------------

if st.session_state.enhanced_results:
    st.divider()
    er = st.session_state.enhanced_results

    st.markdown("### Recruiter Analytics Dashboard")

    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Complete Performance Evaluation")
            m = er["metrics"]
            st.markdown(
                f"<div style='display:flex; flex-wrap:wrap; gap:8px;'>"
                f"<span class='metric-pill'>NDCG Improvement Analysis (NDCG@{m['k']}): <b>{m['ndcg_at_k']}</b></span>"
                f"<span class='metric-pill'>Spearman Correlation Analysis: <b>{m['spearman_r']}</b></span>"
                f"<span class='metric-pill'>Reordered: <b>{m['reordered_pct']}%</b></span>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("#### Metrics History")
            history = ranking_system.get_metrics_history()
            if not history.empty:
                history_display = history.copy()
                history_display["timestamp"] = pd.to_datetime(history_display["timestamp"])
                history_display = history_display.sort_values("timestamp")
                st.line_chart(history_display.set_index("timestamp")[["ndcg_at_k", "spearman_r"]])
            else:
                st.caption("Run enhancement multiple times to see a trend.")

        with col2:
            st.markdown(f"#### Before vs After ranking comparison (Top {m['k']})")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='comparison-card old-rec'>", unsafe_allow_html=True)
                st.markdown("**Old (No Feedback)**")
                for i, cand in enumerate(er["old_candidates"][:10], start=1):
                    st.write(f"{i}. Candidate {cand['candidate_id']} (score={cand.get('final_score', 0):.3f})")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='comparison-card enhanced-rec'>", unsafe_allow_html=True)
                st.markdown("**Enhanced (Feedback-integrated ranking)**")
                for i, cand in enumerate(er["new_candidates"][:10], start=1):
                    st.write(f"{i}. Candidate {cand['candidate_id']} (adj={cand.get('adjusted_score', 0):.3f})")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Detailed Comparison")
        comp_df = er["comparison"].copy()
        st.dataframe(comp_df.head(20))