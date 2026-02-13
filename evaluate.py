import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from recruiter_model import RecruiterRankingSystem

def compute_metrics(relevance_labels, predicted_scores, k=10):
    """
    relevance_labels: Binary list (1 if relevant, 0 otherwise)
    predicted_scores: List of scores for the same items
    """
    # Convert to numpy arrays
    relevance_labels = np.array(relevance_labels)
    predicted_scores = np.array(predicted_scores)

    # Sort indices by predicted scores descending
    indices = np.argsort(predicted_scores)[::-1]

    # Top-K relevance
    top_k_relevance = relevance_labels[indices[:k]]

    # Precision@K
    precision_at_k = np.sum(top_k_relevance) / k

    # NDCG@K
    # ndcg_score expects 2D arrays
    ndcg_at_k = ndcg_score([relevance_labels], [predicted_scores], k=k)

    return precision_at_k, ndcg_at_k

if __name__ == "__main__":
    # Synthetic Evaluation
    print("Running Offline Evaluation...")

    # Example JD
    jd = "Seeking a Python Backend Developer with 5 years of experience in Django and Microservices."

    # Initialize system
    system = RecruiterRankingSystem("data/resumes.csv")

    # Get all candidate scores for this JD
    results = system.rank_candidates(jd, top_k=20)

    # Generate Synthetic Ground Truth
    # In a real scenario, these would be human labels.
    # For this demo, we'll assume candidates with 'Python' and >= 4 years exp are highly relevant (label=1)

    resumes = pd.read_csv("data/resumes.csv")
    relevance = []
    scores = []

    # Map candidate_id to score from results
    candidate_scores = {r['candidate_id']: r['final_score'] for r in results}

    for _, row in resumes.iterrows():
        cid = str(row['candidate_id'])
        # Synthetic relevance label
        is_relevant = 1 if ('Python' in str(row['skills']) and row['experience'] >= 4) else 0
        relevance.append(is_relevant)
        scores.append(candidate_scores.get(cid, 0)) # 0 if not in top_k or filtered

    # Check for NaNs and fill with 0
    scores = np.nan_to_num(scores)

    p10, n10 = compute_metrics(relevance, scores, k=10)

    print(f"Results for JD: '{jd}'")
    print(f"Precision@10: {p10:.4f}")
    print(f"NDCG@10: {n10:.4f}")

    print("\n--- Documentation ---")
    print("Why ranking metrics instead of accuracy?")
    print("Accuracy is used for classification (is this relevant or not). In recruitment, we care about the ORDER.")
    print("A recruiter only looks at the first few candidates. Ranking metrics reward putting the best candidates at the top.")

    print("\nWhy NDCG captures ranking quality better?")
    print("Precision@K only cares if the top K items are relevant, but doesn't care about their order WITHIN the top K.")
    print("NDCG (Normalized Discounted Cumulative Gain) uses a logarithmic discount, meaning that a relevant item")
    print("at position 1 contributes much more than a relevant item at position 10.")
