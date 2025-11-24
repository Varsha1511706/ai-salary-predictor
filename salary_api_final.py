from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import threading
import time

app = FastAPI(
    title="AI Salary Predictor 🇮🇳",
    description="Predict Indian Tech Salaries",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    job_role: str
    experience_years: int
    skills: List[str]
    city: str
    education: str = "Bachelors"
    company_size: str = "Medium"

class PredictionResponse(BaseModel):
    predicted_salary: int
    confidence: float
    salary_range: dict
    currency: str = "INR"
    message: str = ""

# Salary data for Indian market
base_salaries = {
    "Software Engineer": 800000,
    "Data Scientist": 900000,
    "Frontend Developer": 700000,
    "Backend Developer": 750000,
    "Full Stack Developer": 850000,
    "DevOps Engineer": 950000,
    "Product Manager": 1200000,
    "ML Engineer": 1000000,
    "Data Analyst": 600000
}

city_multipliers = {
    "Bangalore": 1.2, "Mumbai": 1.3, "Delhi": 1.25,
    "Hyderabad": 1.1, "Pune": 1.1, "Chennai": 1.0,
    "Gurgaon": 1.3, "Noida": 1.1, "Kolkata": 0.9
}

@app.get("/")
def root():
    return {
        "message": "🇮🇳 AI Salary Predictor API is RUNNING!",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "jobs": "/jobs", 
            "cities": "/cities",
            "skills": "/skills",
            "predict": "/predict"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "salary-predictor"}

@app.get("/jobs")
def get_jobs():
    return {"jobs": list(base_salaries.keys())}

@app.get("/cities")
def get_cities():
    return {"cities": list(city_multipliers.keys())}

@app.get("/skills")
def get_skills():
    skills = ["Python", "Java", "JavaScript", "SQL", "AWS", "Docker", 
              "Machine Learning", "React", "Node.js", "TensorFlow"]
    return {"skills": skills}

@app.post("/predict")
def predict_salary(request: PredictionRequest):
    # Calculate salary
    base = base_salaries.get(request.job_role, 700000)
    multiplier = city_multipliers.get(request.city, 1.0)
    experience_bonus = request.experience_years * 80000
    
    salary = int((base + experience_bonus) * multiplier)
    
    # Adjustments
    if request.education == "Masters":
        salary = int(salary * 1.1)
    elif request.education == "PhD":
        salary = int(salary * 1.2)
    
    if request.company_size == "Large":
        salary = int(salary * 1.15)
    elif request.company_size == "Startup":
        salary = int(salary * 0.9)
    
    # High-value skills bonus
    high_value_skills = ["Machine Learning", "AWS", "Docker", "TensorFlow"]
    user_skills = [skill for skill in request.skills if skill in high_value_skills]
    if user_skills:
        salary = int(salary * (1 + len(user_skills) * 0.05))
    
    message = f"As {request.job_role} with {request.experience_years} years in {request.city}"
    if user_skills:
        message += f". High-value skills: {', '.join(user_skills)}"
    
    return PredictionResponse(
        predicted_salary=salary,
        confidence=0.85,
        salary_range={"min": int(salary * 0.8), "max": int(salary * 1.2)},
        message=message
    )

def start_server():
    print("🚀 STARTING AI SALARY PREDICTOR SERVER...")
    print("📍 Server will run on: http://localhost:8000")
    print("⏳ Starting in 3 seconds...")
    time.sleep(3)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    start_server()
