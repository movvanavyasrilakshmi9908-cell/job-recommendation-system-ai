import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # allow model downloads without SSL errors

import os
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from model import JobRecommendationSystem

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="JobFusion",
    page_icon="💼",
    layout="wide"
)

# -------------------- CONSTANTS --------------------
RATINGS_DIR = os.path.join(os.getcwd(), "data")
RATINGS_PATH = os.path.join(RATINGS_DIR, "ratings.csv")

# -------------------- INITIAL SETUP --------------------
@st.cache_resource
def load_model():
    return JobRecommendationSystem("JobsFE.csv")

recommender = load_model()

# Initialize session state
for key, default in {
    "feedback": [],
    "resume_text": "",
    "resume_id": None,
    "job_results": [],
    "results_ready": False,
    "last_uploaded_file": None,
    "enhanced_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------- HELPERS ---------a-----------
import re

def extract_text_from_pdf(pdf_file):
    """Extract plain text from uploaded PDF resume."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()


def extract_candidate_details(text):
    """Extract basic details from resume text to save for recruiters."""
    # Name: Assume the first non-empty line is the candidate's name
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name = lines[0] if lines else "Unknown"

    # Email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group(0) if email_match else "Unknown"

    # Phone
    phone_match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    phone = phone_match.group(0) if phone_match else "Unknown"

    # Experience (heuristic: look for digits followed by 'years')
    exp_match = re.search(r"(\d+)\s*\+?\s*years?", text, re.IGNORECASE)
    experience = int(exp_match.group(1)) if exp_match else 0

    # Skills (simple keyword matching)
    common_skills = {
        "python", "java", "aws", "docker", "react", "sql", "machine learning",
        "nlp", "kubernetes", "c++", "javascript", "node.js", "django", "flask",
        "pytorch", "tensorflow", "vue.js", "css", "html", "azure", "linux", "oracle"
    }
    found_skills = {s for s in common_skills if s in text.lower()}
    skills_str = ";".join(found_skills) if found_skills else "Unknown"

    return name, email, phone, experience, skills_str


def ensure_ratings_dir():
    """Make sure the data directory exists."""
    os.makedirs(RATINGS_DIR, exist_ok=True)


def save_rating(resume_id, job_id, rating):
    """Save or update numeric rating for (resume_id, job_id)."""
    ensure_ratings_dir()
    feedback_file = RATINGS_PATH
    rating_value = int(rating)

    # Load or create DataFrame
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
    else:
        df = pd.DataFrame(columns=["resume_id", "job_id", "rating"])

    # Normalize dtypes
    if not df.empty:
        df["resume_id"] = df["resume_id"].astype(str)
        df["job_id"] = df["job_id"].astype(str)

    new_entry = {"resume_id": str(resume_id), "job_id": str(job_id), "rating": rating_value}

    mask = (df["resume_id"] == str(resume_id)) & (df["job_id"] == str(job_id))
    if mask.any():
        df.loc[mask, "rating"] = rating_value
        st.toast(f"🔄 Updated rating for job {job_id}")
    else:
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        st.toast(f"💾 Added new rating for job {job_id}")

    df["rating"] = df["rating"].astype(int)
    df.to_csv(feedback_file, index=False)


def save_resume_for_recruiter(resume_text):
    """Saves the uploaded resume to resumes.csv so recruiters can view it."""
    ensure_ratings_dir()
    resumes_file = os.path.join(RATINGS_DIR, "resumes.csv")

    # Extract details
    name, email, phone, experience, skills = extract_candidate_details(resume_text)

    # Check if the CSV exists and determine the next ID
    if os.path.exists(resumes_file):
        df = pd.read_csv(resumes_file)
        next_id = int(df["candidate_id"].max()) + 1 if not df.empty else 1
    else:
        df = pd.DataFrame(columns=["candidate_id", "name", "phone", "email", "resume_text", "skills", "experience"])
        next_id = 1

    # Check if we already have this exact resume text (to avoid duplicates from same session)
    if not df.empty and resume_text in df["resume_text"].values:
        return

    # Create the new entry
    new_entry = {
        "candidate_id": next_id,
        "name": name,
        "phone": phone,
        "email": email,
        "resume_text": resume_text,
        "skills": skills,
        "experience": experience
    }

    # Append to the dataframe and save
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(resumes_file, index=False)
    st.toast("📄 Resume successfully indexed for recruiter visibility!")


def run_recommend(
    source: str = "button",
    location_weight: float = 0.1,
    salary_weight: float = 0.1,
    experience_weight: float = 0.1,
    user_location: str = "New York, NY",
    user_salary: str = "100000",
    user_experience: str = "5"
):
    """Generate personalized job recommendations (uses feedback if available)."""
    with st.spinner("🔍 Analyzing your resume and finding best job matches..."):
        results = recommender.recommend_jobs(
            st.session_state.resume_text,
            top_n=20,
            use_feedback=True,
            location_weight=location_weight,
            salary_weight=salary_weight,
            experience_weight=experience_weight,
            user_location=user_location,
            user_salary=user_salary,
            user_experience=user_experience
        )

    st.session_state.job_results = results["recommended_jobs"]
    st.session_state.resume_quality = results.get("resume_quality", None)
    st.session_state.results_ready = True
    st.session_state.enhanced_results = None
    if source == "main":
        st.success(f"✅ Found {len(st.session_state.job_results)} matching jobs!")


# -------------------- UI THEME --------------------
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
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
st.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0;"></h1>
        <h3 style="margin: 0; margin-left: 10px; color: #555;"></h3>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- LAYOUT --------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("#### Your Profile")
    user_location = st.text_input("Your Location (e.g., city, state)", "Berlin Germany")
    user_salary = st.text_input("Desired Salary (e.g., 120000)", "220000")
    user_experience = st.text_input("Years of Experience", "3")
    st.markdown("---")
    st.markdown("#### Factor Weights")
    location_weight = st.slider("Location", 0.0, 1.0, 0.3, key="location_slider")
    salary_weight = st.slider("Salary", 0.0, 1.0, 0.4, key="salary_slider")
    experience_weight = st.slider("Experience", 0.0, 1.0, 0.3, key="experience_slider")

with right_col:
    # st.markdown("#### Your JobFusion")
    with st.container():
        st.markdown(
            """
            <div class="upload-box">
                <p>Upload your resume (PDF only)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("", type=["pdf"], key="file_uploader", label_visibility="collapsed")

    if uploaded_file is not None:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
            st.session_state.resume_id = hash(st.session_state.resume_text) % (10**8)
            st.session_state.job_results = []
            st.session_state.results_ready = False
            st.session_state.enhanced_results = None
            st.toast("Resume uploaded successfully!")

            # Save the newly uploaded resume to resumes.csv
            save_resume_for_recruiter(st.session_state.resume_text)

    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("Find My Perfect Jobs", disabled=not bool(st.session_state.resume_text)):
            run_recommend(
                "main",
                location_weight=location_weight,
                salary_weight=salary_weight,
                experience_weight=experience_weight,
                user_location=user_location,
                user_salary=user_salary,
                user_experience=user_experience
            )
    with btn_cols[1]:
        if st.button("Enhance with AI Feedback", disabled=not bool(st.session_state.resume_text)):
            with st.spinner("🧠 Retraining model using your feedback..."):
                st.session_state.enhanced_results = recommender.retrain_with_feedback(
                    st.session_state.resume_text, top_n=20
                )

# -------------------- JOB RESULTS --------------------
if st.session_state.results_ready and st.session_state.job_results:
    st.markdown("---")
    st.markdown("### Recommended Jobs for You")

    for i, job in enumerate(st.session_state.job_results[:20], start=1):
        job_id = job.get("Job Id", f"unknown_{i}")
        with st.container():
            st.markdown("<div class='job-card'>", unsafe_allow_html=True)

            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"#### {job['position'].title()} — {job['workplace'].title()}")
            with cols[1]:
                st.markdown(f"<span style='float: right;'>{job['working_mode'].capitalize()}</span>", unsafe_allow_html=True)

            base_sim = float(job.get("similarity", 0))
            adj_sim = float(job.get("adjusted_score", base_sim))
            st.slider("Match Score", 0.0, 1.0, adj_sim, disabled=True, key=f"score_{i}")

            with st.expander("Details"):
                st.write(f"**Duties:** {job['job_role_and_duties'][:250]}...")
                st.write(f"**Skills Required:** {job['requisite_skill']}")
                if job.get("matched_skills"):
                    chips = "".join([f"<span class='tag'>{s.strip()}</span>" for s in str(job['matched_skills']).split(",") if s.strip()][:8])
                    if chips:
                        st.markdown(f"**Matched Skills:** {chips}", unsafe_allow_html=True)

            feedback_cols = st.columns([3, 2])
            with feedback_cols[0]:
                rating = st.slider("Your Rating", 1, 5, 3, key=f"slider_{i}")
            with feedback_cols[1]:
                if st.button("Submit Feedback", key=f"rate_btn_{i}"):
                    save_rating(st.session_state.resume_id, job_id, rating)
                    st.toast(f"⭐ You rated Job {i} as {rating}/5")
                    run_recommend(
                        "rating",
                        location_weight=st.session_state.get('location_slider', 0.2),
                        salary_weight=st.session_state.get('salary_slider', 0.25),
                        experience_weight=st.session_state.get('experience_slider', 0.15),
                    )

            st.markdown("</div>", unsafe_allow_html=True)

# -------------------- ENHANCED MODEL (OLD vs NEW) --------------------
if st.session_state.enhanced_results:
    st.divider()
    er = st.session_state.enhanced_results
    
    st.markdown("### Enhanced Recommendations Dashboard")

    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("####  Ranking Consistency Evaluation")
            m = er["metrics"]
            st.markdown(
                f"<div style='display:flex; flex-wrap:wrap; gap:8px;'>"
                f"<span class='metric-pill'>NDCG@20: <b>{m['ndcg_at_k']}</b></span>"
                f"<span class='metric-pill'>Spearman-R: <b>{m['spearman_r']}</b></span>"
                f"<span class='metric-pill'>Reordered: <b>{m['reordered_pct']}%</b></span>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("####  Metrics ")
            history = recommender.get_metrics_history()
            if not history.empty:
                history_display = history.copy()
                history_display["timestamp"] = pd.to_datetime(history_display["timestamp"])
                history_display = history_display.sort_values("timestamp")
                st.line_chart(history_display.set_index("timestamp")[["ndcg_at_k", "spearman_r"]])
            else:
                st.caption("Run enhancement multiple times to see a trend.")
        
        with col2:
            st.markdown("####  Old vs Enhanced (Top 20)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='comparison-card old-rec'>", unsafe_allow_html=True)
                st.markdown("**Old (No Feedback)**")
                for i, job in enumerate(er["old_jobs"][:10], start=1):
                    st.write(f"{i}. {job['position'].title()} (sim={job.get('similarity', 0):.3f})")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='comparison-card enhanced-rec'>", unsafe_allow_html=True)
                st.markdown("**Enhanced (Feedback)**")
                for i, job in enumerate(er["new_jobs"][:10], start=1):
                    st.write(f"{i}. {job['position'].title()} (adj={job.get('adjusted_score', 0):.3f})")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Detailed Comparison (Top 20)")
        comp_df = er["comparison"].copy()
        st.dataframe(comp_df.head(20))

# -------------------- DEBUG PANEL --------------------
with st.sidebar.expander("🐞 Debug Info"):
    st.write("Ratings file path:", RATINGS_PATH)
    if os.path.exists(RATINGS_PATH):
        st.dataframe(pd.read_csv(RATINGS_PATH, on_bad_lines='skip').tail(5))
    st.write("Session keys:", list(st.session_state.keys()))
