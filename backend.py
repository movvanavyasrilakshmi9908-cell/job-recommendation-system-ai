from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recruiter_model import RecruiterRankingSystem
import uvicorn

app = FastAPI(title="Recruiter Candidate Ranking API")

# Initialize the ranking system
try:
    ranking_system = RecruiterRankingSystem("data/resumes.csv")
except Exception as e:
    print(f"Error initializing ranking system: {e}")
    ranking_system = None

class RankRequest(BaseModel):
    job_description: str
    top_k: int = 20
    min_experience: int = 0

@app.post("/rank_candidates")
async def rank_candidates(request: RankRequest):
    if ranking_system is None:
        raise HTTPException(status_code=500, detail="Ranking system not initialized. Check if resumes.csv exists.")

    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    results = ranking_system.rank_candidates(
        job_description=request.job_description,
        top_k=request.top_k,
        min_experience=request.min_experience
    )

    return {"candidates": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
