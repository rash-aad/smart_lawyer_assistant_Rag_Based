
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import the compatibility module
from . import cohere_utils as llm

app = FastAPI(title="Lawyers Assistant - Ollama Backend")

class TextRequest(BaseModel):
    text: str
    task: Optional[str] = None

@app.post("/summarize")
def summarize(req: TextRequest):
    try:
        out = llm.summarization(req.text)
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/argument-mine")
def argument_mine(req: TextRequest):
    try:
        out = llm.argument_mining(req.text)
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategy-suggestions")
def strategy(req: TextRequest):
    try:
        out = llm.strategy_suggestions(req.text)
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk-prediction")
def risk_prediction(req: TextRequest):
    try:
        out = llm.risk_prediction(req.text)
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/future-steps")
def future_steps(req: TextRequest):
    try:
        out = llm.future_steps(req.text)
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

