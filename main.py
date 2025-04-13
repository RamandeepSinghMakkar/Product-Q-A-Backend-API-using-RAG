import json
import os
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import asyncio

# Load environment variables from .env file if available.
load_dotenv()
SENT_EMBED_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "google/flan-t5-small")

# Setup logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app instance.
app = FastAPI(title="Product Q&A API")

# Mount static files directory to serve the front end.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# Data structures to store product info.
product_chunks = []  # List of dictionaries with keys: product_id, chunk.
embeddings_matrix = None  # A NumPy array to hold embeddings.

# Initialize the embedding and generation models.
embed_model = SentenceTransformer(SENT_EMBED_MODEL)
generator = pipeline("text2text-generation", model=GEN_MODEL)

# Pydantic models for API request/response.
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    primary_product_id: str

def load_and_preprocess_data(filepath: str):
    logger.info("Loading product data from %s", filepath)
    try:
        with open(filepath, "r") as f:
            content = f.read().strip()
            if not content:
                raise ValueError("File is empty.")
            products = json.loads(content)
    except Exception as e:
        logger.error("Error reading JSON file: %s", e)
        raise e

    chunks = []
    for product in products:
        # Join product fields into one chunk.
        text = f"{product['name']}: {product['description']} Features: {product['features']}"
        chunks.append({
            "product_id": product["product_id"],
            "chunk": text
        })
    return chunks

def generate_embeddings_for_chunks(chunks):
    texts = [item["chunk"] for item in chunks]
    logger.info("Generating embeddings for product chunks...")
    embeddings = embed_model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    return embeddings

def build_vector_store(chunks, embeddings):
    global embeddings_matrix, product_chunks
    embeddings_matrix = np.array(embeddings)
    product_chunks = chunks
    logger.info("Built vector store with %d items.", len(chunks))

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_top_chunk(query_embedding):
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings_matrix]
    # Get the index of the most similar embedding.
    top_index = int(np.argmax(similarities))
    top_chunk = {
        "product_id": product_chunks[top_index]["product_id"],
        "chunk": product_chunks[top_index]["chunk"],
        "similarity": similarities[top_index]
    }
    logger.info("Top product ID: %s", top_chunk["product_id"])
    return top_chunk

def construct_prompt(question, top_chunk):
    context_text = f"Product ID: {top_chunk['product_id']}\nContext: {top_chunk['chunk']}"
    prompt = (
        "Answer the following question using ONLY the context provided below. "
        "Mention the product_id if it is used in your answer. "
        "If the context does not provide enough information, say so.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )
    return prompt

async def query_llm(prompt):
    try:
        result = await asyncio.to_thread(generator, prompt, max_length=150, do_sample=False)
        answer = result[0]['generated_text'].strip()
        return answer
    except Exception as e:
        logger.error("Error querying LLM: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM API error: {e}")

@app.on_event("startup")
def startup_event():
    try:
        chunks = load_and_preprocess_data("products.json")
        embeddings = generate_embeddings_for_chunks(chunks)
        build_vector_store(chunks, embeddings)
        logger.info("Startup completed successfully.")
    except Exception as e:
        logger.error("Startup error: %s", e)
        raise e

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    question = request.question
    # Generate the embedding for the query.
    query_embedding = embed_model.encode(question, convert_to_tensor=False)
    top_chunk = retrieve_top_chunk(query_embedding)
    prompt = construct_prompt(question, top_chunk)
    logger.info("Constructed prompt: %s", prompt)
    answer = await query_llm(prompt)

    # Directly return the top product's ID.
    primary_product_id = top_chunk["product_id"]

    return AnswerResponse(answer=answer, primary_product_id=primary_product_id)

# To run the server using uvicorn:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
