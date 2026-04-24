from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

from inference import predict, needs_clarification

app = FastAPI(title="Udyam Multi-Code Predictor API")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = None
    clarification: Optional[Dict[str, Any]] = None

@app.post("/api/predict", response_model=PredictResponse)
def api_predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        return PredictResponse(status="error")

    # Get initial prediction for confidence check
    quick_result = predict(text, top_k=1)
    best_conf = quick_result["top_nics"][0]["confidence"]

    # Check if clarification is needed
    clarify = needs_clarification(text, best_conf)
    
    if clarify:
        return PredictResponse(
            status="clarification_needed",
            clarification=clarify
        )
    else:
        # Full prediction
        result = predict(text, top_k=3)
        return PredictResponse(
            status="success",
            result=result
        )

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
