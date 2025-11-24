from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="AI Salary Predictor 🇮🇳",
    description="Predict Indian tech salaries using machine learning",
    version="1.0.0"
)

# Request model
class PredictionRequest(BaseModel):
    job_role: str
    experience_years: int
    skills: List[str]
    city: str
    education: str = "Bachelors"
    company_size: str = "Medium"

# Response model
class PredictionResponse(BaseModel):
    predicted_salary: int
    confidence: float
    salary_range: dict
    currency: str = "INR"
    message: str = ""

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "🇮🇳 AI Salary Prediction API is running!",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "jobs": "/api/v1/jobs",
            "cities": "/api/v1/cities",
            "skills": "/api/v1/skills", 
            "predict": "/api/v1/predict"
        }
    }

# Health endpoint
@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "salary-predictor",
        "version": "1.0.0"
    }

# Jobs endpoint
@app.get("/api/v1/jobs")
def get_jobs():
    jobs = [
        "Software Engineer", "Data Scientist", "Frontend Developer",
        "Backend Developer", "Full Stack Developer", "DevOps Engineer",
        "Product Manager", "ML Engineer", "Data Analyst"
    ]
    return {"jobs": sorted(jobs)}

# Cities endpoint
@app.get("/api/v1/cities")
def get_cities():
    cities = [
        "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune",
        "Chennai", "Gurgaon", "Noida", "Kolkata"
    ]
    return {"cities": sorted(cities)}

# Skills endpoint
@app.get("/api/v1/skills")
def get_skills():
    skills = [
        "Python", "Java", "JavaScript", "SQL", "AWS",
        "Docker", "Machine Learning", "React", "Node.js",
        "TensorFlow", "PyTorch", "Data Analysis"
    ]
    return {"skills": sorted(skills)}

# Prediction endpoint
@app.post("/api/v1/predict")
def predict_salary(request: PredictionRequest):
    # Salary calculation logic
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
    
    # Calculate base salary
    base = base_salaries.get(request.job_role, 700000)
    multiplier = city_multipliers.get(request.city, 1.0)
    experience_bonus = request.experience_years * 80000
    
    salary = int((base + experience_bonus) * multiplier)
    
    # Adjust for education
    if request.education == "Masters":
        salary = int(salary * 1.1)
    elif request.education == "PhD":
        salary = int(salary * 1.2)
    
    # Adjust for company size
    if request.company_size == "Large":
        salary = int(salary * 1.15)
    elif request.company_size == "Startup":
        salary = int(salary * 0.9)
    
    # High-value skills bonus
    high_value_skills = ["Machine Learning", "AWS", "Docker", "TensorFlow"]
    user_high_value_skills = [skill for skill in request.skills if skill in high_value_skills]
    if user_high_value_skills:
        salary = int(salary * (1 + len(user_high_value_skills) * 0.05))
    
    return PredictionResponse(
        predicted_salary=salary,
        confidence=0.85,
        salary_range={
            "min": int(salary * 0.8),
            "max": int(salary * 1.2)
        },
        message=f"Based on {request.experience_years} years as {request.job_role} in {request.city}"
    )

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
