# Recruiter Model Process Flow

The recruiter model evaluates and ranks candidates based on their resumes against a given Job Description (JD). Below is the complete process flow of the system, detailing each step from input to the final candidate ranking.

## 1. Job Description Input
- The system receives the Job Description (JD) text either through direct text input or by extracting text from an uploaded PDF file via the Streamlit dashboard (`recruiter_app.py`).

## 2. Resume Collection and Parsing
- Resumes are collected and loaded from a CSV file (`resumes.csv`) during the initialization of the `RecruiterRankingSystem`.
- The dataset contains candidate details such as candidate IDs, resume text, extracted skills, and years of experience.

## 3. Text Preprocessing
- Both the Job Description and the parsed resume texts undergo a cleaning process (`clean_text` method).
- The text is converted to lowercase, punctuation is removed, and extra whitespace is stripped to ensure consistency before embedding generation.

## 4. Embedding Generation
- The system uses a pre-trained SentenceTransformer model (`paraphrase-MiniLM-L6-v2`) to generate dense vector embeddings for both the cleaned resume texts and the Job Description.
- Embeddings are normalized (L2 normalization) to allow for accurate Cosine Similarity computation.

## 5. Vector Indexing using FAISS
- The normalized resume embeddings are added to a FAISS (Facebook AI Similarity Search) index (`IndexFlatIP`).
- This index facilitates highly efficient Inner Product (Cosine Similarity) searches across the candidate pool.

## 6. Similarity Search
- The normalized embedding of the cleaned Job Description is queried against the FAISS index to find the most semantically similar resumes.
- The system retrieves a baseline semantic score (Cosine Similarity) and candidate indices from the FAISS search.

## 7. Hybrid Scoring Calculation
- The system extracts specific required skills and minimum experience years from the Job Description using regular expressions.
- It computes three individual scores for each candidate:
  - **Semantic Score**: Derived directly from the FAISS similarity search.
  - **Skill Score**: The ratio of JD-required skills found in the candidate's resume compared to the total required skills.
  - **Experience Score**: Calculated based on the candidate's experience relative to the minimum experience required by the JD (capped at 1.0).
- A final hybrid score is computed using a weighted sum:
  `Final Score = (0.5 * Semantic Score) + (0.3 * Skill Score) + (0.2 * Experience Score)`.

## 8. Candidate Ranking
- Candidates are sorted in descending order based on their `Final Score`.
- The top *K* candidates (configurable via the UI) are returned.
- The Streamlit app displays the ranked candidates, including their individual scores, matched skills, experience, and a brief resume summary.
