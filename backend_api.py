import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import time

# Import your inference pipeline (DO NOT change it)
from inference_pipeline import analyze_patrol_note

# ================= FASTAPI APP =================
app = FastAPI(
    title="Municipal Parking Violation NLP API",
    description="NER + Classification API for Patrol Notes Analysis",
    version="1.0.0"
)

# ================= CORS MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= REQUEST / RESPONSE SCHEMAS =================
class PatrolNoteRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    input_text: str
    entities: Dict[str, List[str]]
    violation_category: str
    violation_subtype: str
    severity: str
    inference_time_ms: float

# ================= HEALTH CHECK =================
@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "MPVPN NLP Backend is live"
    }

# ================= ANALYSIS ENDPOINT =================
@app.post("/analyze", response_model=AnalysisResponse)
def analyze_note(request: PatrolNoteRequest):
    """
    Analyze a patrol note:
    - Named Entity Recognition (NER)
    - Violation Category Classification
    - Severity Prediction
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    start_time = time.time()

    try:
        result = analyze_patrol_note(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    inference_time = round((time.time() - start_time) * 1000, 2)

    return {
        "input_text": result["input_text"],
        "entities": result["entities"],
        "violation_category": result["violation_category"],
        "violation_subtype": result["violation_subtype"],
        "severity": result["severity"],
        "inference_time_ms": inference_time
    }

# ================= OPTIONAL: BATCH ANALYSIS =================
class BatchRequest(BaseModel):
    texts: List[str]

@app.post("/analyze-batch")
def analyze_batch(request: BatchRequest):
    """
    Analyze multiple patrol notes at once
    """
    results = []

    for text in request.texts:
        if not text.strip():
            continue
        results.append(analyze_patrol_note(text))

    return {
        "count": len(results),
        "results": results
    }

# ================= MAIN =================
if __name__ == "__main__":
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
