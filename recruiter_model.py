import os
import re
import string
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- MODEL LOADING ----------------

_model = None

def get_model():
    """
    Lazy load the SentenceTransformer model.
    This prevents multiple downloads and avoids Streamlit reload issues.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")
    return _model


# ---------------- RANKING SYSTEM ----------------

class RecruiterRankingSystem:

    def __init__(self, resumes_csv: str):

        if not os.path.exists(resumes_csv):
            raise FileNotFoundError(f"Resumes CSV not found: {resumes_csv}")

        self.model = get_model()

        self.resumes_df = pd.read_csv(resumes_csv)

        self.resume_texts = self.resumes_df["resume_text"].astype(str).tolist()

        # ---------------- EMBEDDINGS ----------------

        resume_embeddings = self.model.encode(
            self.resume_texts,
            convert_to_numpy=True
        ).astype(np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(resume_embeddings, axis=1, keepdims=True)
        self.resume_embeddings = resume_embeddings / (norms + 1e-9)

        # ---------------- FAISS INDEX ----------------

        self.dim = self.resume_embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(self.resume_embeddings)

    # ---------------- TEXT CLEANING ----------------

    def clean_text(self, text: str):

        text = text.lower()

        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )

        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ---------------- SKILL EXTRACTION ----------------

    def extract_skills(self, text: str):

        common_skills = {
            "python","django","flask","fastapi","java","spring boot",
            "react","angular","vue.js","javascript","node.js",
            "typescript","c++","embedded","aws","azure","docker",
            "kubernetes","postgresql","mysql","oracle","redis",
            "mongodb","machine learning","ai","nlp","pytorch",
            "tensorflow","scikit-learn","ci/cd","linux",
            "cloud architecture","figma","adobe xd","ui/ux",
            "tableau","sql","excel","spark","hadoop","scala",
            "solidity","ethereum","blockchain"
        }

        text_lower = text.lower()

        found_skills = {
            skill for skill in common_skills
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower)
        }

        return found_skills

    # ---------------- EXPERIENCE EXTRACTION ----------------

    def extract_experience_years(self, text: str):

        matches = re.findall(r"(\d+)\s*\+?\s*years?", text.lower())

        if matches:
            return max([int(m) for m in matches])

        return 0

    # ---------------- MAIN RANKING FUNCTION ----------------

    def rank_candidates(
        self,
        job_description: str,
        top_k: int = 20,
        min_experience: int = 0
    ):

        clean_jd = self.clean_text(job_description)

        jd_embedding = self.model.encode(
            [clean_jd],
            convert_to_numpy=True
        ).astype(np.float32)

        jd_embedding = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-9)

        # -------- FAISS SEARCH --------

        distances, indices = self.index.search(
            jd_embedding,
            len(self.resume_texts)
        )

        jd_skills = self.extract_skills(clean_jd)

        jd_required_exp = self.extract_experience_years(clean_jd)

        results = []

        for i, idx in enumerate(indices[0]):

            candidate = self.resumes_df.iloc[idx]

            semantic_score = float(distances[0][i])

            resume_skills = set(
                str(candidate["skills"]).lower().split(";")
            )

            # -------- SKILL SCORE --------

            if not jd_skills:
                skill_score = 1.0
                matched_skills = set()
            else:
                matched_skills = jd_skills.intersection(resume_skills)
                skill_score = len(matched_skills) / len(jd_skills)

            # -------- EXPERIENCE SCORE --------

            candidate_exp = float(candidate["experience"])

            if candidate_exp < min_experience:
                continue

            if jd_required_exp == 0:
                experience_score = 1.0
            else:
                experience_score = min(candidate_exp / jd_required_exp, 1.0)

            # -------- FINAL SCORE --------

            final_score = (
                0.5 * semantic_score +
                0.3 * skill_score +
                0.2 * experience_score
            )

            results.append({

                "candidate_id": str(candidate["candidate_id"]),

                "final_score": round(final_score, 4),

                "semantic_score": round(semantic_score, 4),

                "skill_score": round(skill_score, 4),

                "experience_score": round(experience_score, 4),

                "matched_skills": sorted(list(matched_skills)),

                "all_skills": sorted(list(resume_skills)),

                "experience": candidate_exp,

                "resume_summary":
                candidate["resume_text"][:200] + "..."
            })

        results = sorted(
            results,
            key=lambda x: x["final_score"],
            reverse=True
        )

        return results[:top_k]