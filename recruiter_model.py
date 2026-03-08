import os
import re
import string
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

def _spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Spearman rank correlation without external deps.
    Returns in [-1, 1]. Handles constant arrays.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    a_ranks = pd.Series(a).rank(method="average").to_numpy()
    b_ranks = pd.Series(b).rank(method="average").to_numpy()
    a_center = a_ranks - a_ranks.mean()
    b_center = b_ranks - b_ranks.mean()
    denom = (np.linalg.norm(a_center) * np.linalg.norm(b_center)) + 1e-9
    return float(np.dot(a_center, b_center) / denom)


def _ensure_data_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


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

        # Metrics log path
        self.metrics_path = os.path.join("data", "recruiter_metrics.csv")

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

    def load_feedback_embeddings(self, jd_id: str, feedback_file: str = "data/recruiter_ratings.csv"):
        """
        Load ratings and return (embeddings, ratings, merged_df).
        - ratings must be numeric (1..5)
        """
        if not os.path.exists(feedback_file):
            return None, None, None

        df = pd.read_csv(feedback_file)
        if df.empty or "rating" not in df.columns:
            return None, None, None

        # Filter by jd_id
        df = df[df["jd_id"].astype(str) == str(jd_id)]
        if df.empty:
            return None, None, None

        # Ensure numeric ratings and string ids for join safety
        df = df[df["rating"].astype(str).str.isnumeric()].copy()
        if df.empty:
            return None, None, None
        df["rating"] = df["rating"].astype(float)

        # Join with resume info to build embeddings for rated resumes
        merged = pd.merge(df, self.resumes_df, left_on="candidate_id", right_on="candidate_id", how="inner")
        if merged.empty:
            return None, None, None

        resume_embeds = self.model.encode(merged["resume_text"].astype(str).tolist(), convert_to_numpy=True).astype(np.float32)
        ratings = merged["rating"].to_numpy(dtype=np.float32)
        return resume_embeds, ratings, merged

    @staticmethod
    def _alpha_from_num_ratings(n_ratings: int) -> float:
        """
        Adaptive blend between JD similarity (alpha) and feedback (1-alpha).
        - Start at alpha≈0.9 with few ratings; decay to 0.5 as ratings grow.
        """
        alpha = 0.9 - 0.04 * n_ratings  # each rating reduces JD weight by 0.04
        return float(np.clip(alpha, 0.5, 0.9))

    # ---------------- MAIN RANKING FUNCTION ----------------

    def rank_candidates(
        self,
        job_description: str,
        top_k: int = 20,
        min_experience: int = 0,
        use_feedback: bool = False,
        jd_id: str = None
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


        # Base final score calculation
        for r in results:
            r["adjusted_score"] = r["final_score"]

        if use_feedback and jd_id:
            rated_embeds, ratings, _ = self.load_feedback_embeddings(jd_id=jd_id)
            if rated_embeds is not None and len(ratings) > 0:
                # Weighted user preferences vector based on feedback
                if np.max(ratings) == np.min(ratings):
                    norm_r = np.ones_like(ratings) / len(ratings)
                else:
                    norm_r = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-9)

                if np.sum(norm_r) == 0:
                    norm_r = np.ones_like(ratings) / len(ratings)

                user_vec = np.average(rated_embeds, axis=0, weights=norm_r).astype(np.float32)

                # Apply feedback-driven re-ranking
                alpha = self._alpha_from_num_ratings(len(ratings))          # jd weight
                beta = 1.0 - alpha                                          # feedback weight

                for r in results:
                    candidate_id = r["candidate_id"]
                    candidate_idx = self.resumes_df[self.resumes_df["candidate_id"].astype(str) == candidate_id].index[0]
                    cand_embed = self.resume_embeddings[candidate_idx]

                    # Cosine similarity to user preference vector
                    up_sim = np.dot(cand_embed, user_vec) / (
                        np.linalg.norm(cand_embed) * np.linalg.norm(user_vec) + 1e-9
                    )

                    r["adjusted_score"] = alpha * r["final_score"] + beta * up_sim

        # Sort by adjusted_score if use_feedback, otherwise final_score
        sort_key = "adjusted_score" if use_feedback else "final_score"
        results = sorted(
            results,
            key=lambda x: x[sort_key],
            reverse=True
        )

        return results[:top_k]

    # ---------------- RETRAIN / EVALUATE ----------------

    def retrain_with_feedback(self, job_description: str, jd_id: str, top_k: int = 20, min_experience: int = 0):
        """
        Compare old vs enhanced results and compute metrics.
        """
        # Old: no feedback
        old_results = self.rank_candidates(job_description, top_k=top_k, min_experience=min_experience, use_feedback=False)
        old = pd.DataFrame(old_results)

        # New: with feedback
        new_results = self.rank_candidates(job_description, top_k=top_k, min_experience=min_experience, use_feedback=True, jd_id=jd_id)
        new = pd.DataFrame(new_results)

        if old.empty or new.empty:
             return None

        # Align on candidate_id to compare scores directly
        comp = old[["candidate_id", "final_score"]].merge(
            new[["candidate_id", "adjusted_score"]], on="candidate_id", how="outer"
        )

        # Metrics (fill NAs with 0 for fair comparison)
        y_true = comp["final_score"].fillna(0).to_numpy()
        y_score = comp["adjusted_score"].fillna(0).to_numpy()

        # NDCG@top_k
        ndcg = float(ndcg_score([y_true], [y_score]))

        # Spearman rank correlation
        spear = _spearman_r(y_true, y_score)

        # Reordered percentage (by candidate_id order change)
        old_order = {cid: i for i, cid in enumerate(old["candidate_id"].tolist())}
        new_order = {cid: i for i, cid in enumerate(new["candidate_id"].tolist())}
        common_ids = [cid for cid in old_order if cid in new_order]
        moved = sum(1 for cid in common_ids if old_order[cid] != new_order[cid])
        reordered_pct = (moved / max(1, len(common_ids))) * 100.0

        # Log metrics over time
        _ensure_data_dir(self.metrics_path)
        log_row = pd.DataFrame([{
            "timestamp": pd.Timestamp.now().isoformat(),
            "jd_id": jd_id,
            "ndcg_at_k": round(ndcg, 4),
            "spearman_r": round(float(spear), 4),
            "reordered_pct": round(reordered_pct, 2),
            "k": top_k
        }])
        if os.path.exists(self.metrics_path):
            log_row.to_csv(self.metrics_path, mode="a", header=False, index=False)
        else:
            log_row.to_csv(self.metrics_path, mode="w", header=True, index=False)

        return {
            "old_candidates": old.to_dict(orient="records"),
            "new_candidates": new.to_dict(orient="records"),
            "comparison": comp,
            "metrics": {
                "ndcg_at_k": round(ndcg, 3),
                "spearman_r": round(float(spear), 3),
                "reordered_pct": round(reordered_pct, 1),
            },
        }

    def get_metrics_history(self):
        """Return metrics history DataFrame if available."""
        full_schema = [
            "timestamp", "jd_id", "ndcg_at_k", "spearman_r", "reordered_pct", "k"
        ]
        if not os.path.exists(self.metrics_path):
            return pd.DataFrame(columns=full_schema)
        try:
            df = pd.read_csv(self.metrics_path, header=None)
            # Assign columns based on the number of columns found in the file
            num_cols = len(df.columns)
            df.columns = full_schema[:num_cols]
            # Ensure all columns from the schema are present
            for col in full_schema:
                if col not in df.columns:
                    df[col] = pd.NA
            return df[full_schema]
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=full_schema)