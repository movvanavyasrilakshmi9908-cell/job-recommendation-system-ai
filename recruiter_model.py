import os
import re
import string
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score


_model=None

def get_model():
    global _model
    if _model is None:
        _model=SentenceTransformer("paraphrase-MiniLM-L6-v2",device="cpu")
    return _model


class RecruiterRankingSystem:

    def __init__(self,resumes_csv):

        self.model=get_model()

        self.resumes_df=pd.read_csv(resumes_csv)

        self.resume_texts=self.resumes_df["resume_text"].astype(str).tolist()

        emb=self.model.encode(self.resume_texts,convert_to_numpy=True).astype(np.float32)

        emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-9)

        self.resume_embeddings=emb

        self.index=faiss.IndexFlatIP(emb.shape[1])

        self.index.add(emb)

        self.metrics_path="data/recruiter_metrics.csv"


    def clean_text(self,text):

        text=text.lower()

        text=text.translate(str.maketrans("","",string.punctuation))

        return re.sub(r"\s+"," ",text)


    def extract_skills(self,text):

        skills={"python","java","aws","docker","react","sql","machine learning","nlp","kubernetes"}

        return {s for s in skills if s in text}


    def extract_experience_years(self,text):

        m=re.findall(r"(\d+)\s*\+?\s*years?",text)

        return max(map(int,m)) if m else 0


    def rank_candidates(self,job_description,top_k=20,min_experience=0,use_feedback=False,jd_id=None):

        jd=self.clean_text(job_description)

        jd_emb=self.model.encode([jd],convert_to_numpy=True).astype(np.float32)

        jd_emb=jd_emb/(np.linalg.norm(jd_emb)+1e-9)

        dist,idx=self.index.search(jd_emb,len(self.resume_texts))

        jd_skills=self.extract_skills(jd)

        jd_exp=self.extract_experience_years(jd)

        results=[]

        for i,id_ in enumerate(idx[0]):

            cand=self.resumes_df.iloc[id_]

            semantic=float(dist[0][i])

            skills=set(str(cand["skills"]).lower().split(";"))

            match=jd_skills.intersection(skills)

            skill_score=len(match)/len(jd_skills) if jd_skills else 1

            exp=float(cand["experience"])

            if exp<min_experience:
                continue

            exp_score=min(exp/jd_exp,1) if jd_exp else 1

            final=(0.5*semantic)+(0.3*skill_score)+(0.2*exp_score)

            results.append({

                "candidate_id":str(cand["candidate_id"]),

                "final_score":round(final,4),

                "semantic_score":round(semantic,4),

                "skill_score":round(skill_score,4),

                "experience_score":round(exp_score,4),

                "matched_skills":list(match),

                "experience":exp,

                "resume_summary":cand["resume_text"][:200]+"..."
            })

        results=sorted(results,key=lambda x:x["final_score"],reverse=True)

        return results[:top_k]


    def retrain_with_feedback(self,job_description,jd_id,top_k=20,min_experience=0):

        old=self.rank_candidates(job_description,top_k,min_experience)

        new=self.rank_candidates(job_description,top_k,min_experience)

        old_df=pd.DataFrame(old)

        new_df=pd.DataFrame(new)

        comp=old_df.merge(new_df,on="candidate_id",suffixes=("_old","_new"))

        y_true=comp["final_score_old"].values

        y_pred=comp["final_score_new"].values

        ndcg=float(ndcg_score([y_true],[y_pred]))

        spear=np.corrcoef(y_true,y_pred)[0,1]

        return {

            "old_candidates":old,

            "new_candidates":new,

            "comparison":comp,

            "metrics":{

                "ndcg_at_k":round(ndcg,3),

                "spearman_r":round(float(spear),3),

                "reordered_pct":0,

                "k":top_k
            }
        }