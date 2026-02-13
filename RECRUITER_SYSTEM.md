# Recruiter Candidate Ranking System - Documentation

## AI Service (Core Intelligence)

### 1. Model Choice: `paraphrase-MiniLM-L6-v2`
- **Why this model?**
  - It is a lightweight yet powerful SentenceTransformer model.
  - It provides a great balance between speed and performance, making it ideal for real-time ranking applications.
  - It is specifically trained for paraphrasing and semantic similarity, which matches our use case of comparing Job Descriptions to Resumes.
- **Embedding Dimension:** 384
- **Why semantic representation is important?**
  - Keyword matching (TF-IDF/Regex) often fails when different terms are used for the same concept (e.g., "Software Engineer" vs "Full Stack Developer" or "Python Specialist").
  - Semantic embeddings capture the *context* and *intent* of the text, allowing the system to find relevant candidates even if they don't use the exact same keywords as the JD.

### 2. Multi-Factor Ranking Formula
The final score is a weighted combination of three components:
- **Semantic Score (50%):** Measures the conceptual overlap between the JD and the Resume using cosine similarity of embeddings.
- **Skill Match Score (30%):** Calculated as `|intersection(JD_skills, Resume_skills)| / |JD_skills|`. This ensures that explicit technical requirements are met.
- **Experience Score (20%):** Compares the candidate's years of experience against the JD requirement.
  - If `candidate_exp >= required_exp`, score = 1.0
  - Else, score = `candidate_exp / required_exp`

### 3. Evaluation Metrics
- **Precision@10:** Measures what percentage of the top 10 recommended candidates are actually relevant.
- **NDCG@10 (Normalized Discounted Cumulative Gain):** Measures the quality of the ranking by rewarding relevant candidates appearing higher in the list.
- **Why use these?**
  - Recruitment is a ranking problem, not just a classification problem.
  - Recruiter time is valuable; we want the absolute best matches to appear first.

## UI Side (Recruiter Dashboard)

The dashboard provides an intuitive interface for recruiters to:
1. Paste or upload a Job Description.
2. Filter by Top-K and Minimum Experience.
3. View a ranked list of candidates with a detailed score breakdown.
4. Deep dive into candidate details (matched skills, all skills, experience, and resume summary).

## Setup & Running

1. **Start the Backend API:**
   ```bash
   python backend.py
   ```
2. **Start the Recruiter Dashboard:**
   ```bash
   streamlit run recruiter_app.py
   ```
3. **Run Evaluation:**
   ```bash
   python evaluate.py
   ```
