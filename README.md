# Product Q&A Backend API using RAG (Retrieval-Augmented Generation)

This project implements a FastAPI-based backend API that answers user questions about a specific set of products using the **Retrieval-Augmented Generation (RAG)** approach.

It performs:
- Local data loading & preprocessing from a JSON file.
- Embedding generation using a local Sentence Transformer (`all-MiniLM-L6-v2`).
- In-memory vector search using cosine similarity.
- Answer generation using a locally loaded LLM (`google/flan-t5-small`).
- Exposes a REST API endpoint via FastAPI.

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/RamandeepSinghMakkar/Product-Q-A-Backend-API-using-RAG
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```
3. Install Dependencies
 ```bash
pip install -r requirements.txt
```
4. Set Up Environment Variables
Create a .env file in the root directory with the following:

SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
GEN_MODEL=google/flan-t5-small
TOP_K=3


### How to Run the Application
```bash
uvicorn main:app --reload
```
The server will start at http://127.0.0.1:8000.

 Design Choices

🔹 Chunking Strategy:
Product descriptions and features are combined per product.
Texts longer than ~256 tokens are split by sentence using nltk.sent_tokenize(), grouping into chunks under the token limit.
Justification: Helps retain semantic meaning and works well with sentence transformers and FLAN-T5 input limits.
🔹 Embedding Strategy:
Used all-MiniLM-L6-v2 from sentence-transformers for efficient, small-size embeddings.
Embeddings are generated once at startup and stored in-memory (NumPy array).
🔹 Similarity Metric:
Cosine similarity computed manually using np.dot() and vector norms.
Top-K (default = 3) most relevant chunks retrieved based on cosine score.
🔹 LLM Prompting:
google/flan-t5-small is used via HuggingFace's pipeline.
Prompts include:
Retrieved context
Original user question
Clear instructions to answer using ONLY the context, and return product IDs if applicable
🔹 Async & Logging:
FastAPI async endpoints used for non-blocking execution.
Python's logging module used to track:
Incoming questions
Retrieved chunk product IDs
Final generated answer


Example API requests and responses: Check

