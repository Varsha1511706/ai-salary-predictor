from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# Create the app
app = FastAPI(title="AI Salary Predictor 🇮🇳")

# Request model
class PredictionRequest(BaseModel):
    job_role: str
    experience_years: int
    skills: List[str]
    city: str

# ALL ENDPOINTS IN ONE FILE
@app.get("/")
def root():
    return {"message": "API is working! 🇮🇳"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/jobs")
def jobs():
    return {"jobs": ["Software Engineer", "Data Scientist"]}

@app.get("/cities") 
def cities():
    return {"cities": ["Bangalore", "Mumbai", "Delhi"]}

@app.get("/skills")
def skills():
    return {"skills": ["Python", "Java", "SQL"]}

@app.post("/predict")
def predict(request: PredictionRequest):
    salary = 1000000 + (request.experience_years * 100000)
    return {
        "predicted_salary": salary,
        "currency": "INR",
        "message": f"Salary for {request.job_role} in {request.city}"
    }

# RUN THE SERVER
if __name__ == "__main__":
    print("🚀 Starting server on http://localhost:8000")
    print("✅ Endpoints: /, /health, /jobs, /cities, /skills, /predict")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
