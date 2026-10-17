from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# Initialize FastAPI
app = FastAPI(
    title="DocuMind AI API",
    description="Backend API for intelligent document search",
    version="1.0.0"
)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:4b"


# Request model
class QuestionRequest(BaseModel):
    question: str


# Response model
class AnswerResponse(BaseModel):
    answer: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to DocuMind AI!",
        "status": "running"
    }


# AI question-answering endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):

    payload = {
        "model": MODEL_NAME,
        "prompt": request.question,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 500
        }
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:

            response = await client.post(
                OLLAMA_URL,
                json=payload
            )

            response.raise_for_status()

            data = response.json()

            # Get generated response
            answer = data.get("response", "")

            # Diagnostic information
            print("Thinking field present:", bool(data.get("thinking")))
            print("Thinking tag present:", "</think>" in answer)

            # Remove thinking content if it appears in the response
            if "</think>" in answer:
                answer = answer.split("</think>", 1)[1].strip()

            # Check whether the model returned an answer
            if not answer:
                raise HTTPException(
                    status_code=502,
                    detail="Ollama returned an empty answer."
                )

            return AnswerResponse(answer=answer)

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not communicate with Ollama: {error}"
        )