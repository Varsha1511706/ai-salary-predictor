from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="AI Salary Predictor 🇮🇳 🤖",
    description="ML-powered salary predictions for Indian tech roles",
    version="2.0.0"
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
    model_used: str = "XGBoost"

class MLSalaryPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor_data = None
        self.is_ml_loaded = False
        self.load_ml_model()
    
    def load_ml_model(self):
        """Load the trained XGBoost model and preprocessor"""
        try:
            # Load the trained model
            model_path = "ml_pipeline/models/salary_model.pkl"
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the preprocessor
            preprocessor_path = "ml_pipeline/models/preprocessor.pkl"
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor_data = pickle.load(f)
            
            self.is_ml_loaded = True
            print("✅ ML Model loaded successfully!")
            print(f"📊 Features: {len(self.preprocessor_data['feature_columns'])}")
            
        except Exception as e:
            print(f"❌ ML Model loading failed: {e}")
            self.is_ml_loaded = False
    
    def predict_with_ml(self, request: PredictionRequest):
        """Make prediction using the trained ML model"""
        try:
            # Prepare features for ML model
            input_features = self.prepare_ml_features(request)
            
            # Make prediction
            prediction = self.model.predict(input_features)[0]
            
            # Ensure reasonable bounds
            prediction = max(200000, min(5000000, prediction))
            
            return int(prediction)
            
        except Exception as e:
            print(f"ML Prediction error: {e}")
            return None
    
    def prepare_ml_features(self, request: PredictionRequest):
        """Prepare features in the exact format used during training"""
        # Create base data
        data = {
            'job_role': [request.job_role],
            'city': [request.city],
            'experience_years': [request.experience_years],
            'experience_level': self.get_experience_level(request.experience_years),
            'education': [request.education],
            'company_size': [request.company_size],
            'skills': [', '.join([s.lower() for s in request.skills])]
        }
        
        df = pd.DataFrame(data)
        
        # Apply the same feature engineering as during training
        df = self.apply_ml_feature_engineering(df)
        
        # Ensure all training features are present
        for feature in self.preprocessor_data['feature_columns']:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select and order features exactly as training
        feature_df = df[self.preprocessor_data['feature_columns']]
        
        return feature_df
    
    def apply_ml_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the exact same feature engineering as during training"""
        # Experience features
        df['experience_squared'] = df['experience_years'] ** 2
        
        # Metro city indicator
        metro_cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']
        df['is_metro'] = df['city'].apply(lambda x: 1 if x in metro_cities else 0)
        
        # Encode categorical variables
        categorical_columns = ['job_role', 'city', 'experience_level', 'education', 'company_size']
        
        for col in categorical_columns:
            if col in self.preprocessor_data['label_encoders']:
                le = self.preprocessor_data['label_encoders'][col]
                if df[col].iloc[0] in le.classes_:
                    df[f'{col}_encoded'] = le.transform(df[col])
                else:
                    # Handle unseen labels - use most common encoding
                    df[f'{col}_encoded'] = 0
        
        # Add skill features (match training exactly)
        skills_to_check = ['python', 'java', 'sql', 'aws', 'docker', 'machine learning', 'javascript', 'react']
        for skill in skills_to_check:
            df[f'skill_{skill}'] = df['skills'].apply(lambda x: 1 if skill in str(x).lower() else 0)
        
        return df
    
    def get_experience_level(self, experience_years: int) -> str:
        """Convert experience years to level (must match training)"""
        if experience_years <= 2:
            return "Junior"
        elif experience_years <= 5:
            return "Mid"
        elif experience_years <= 10:
            return "Senior"
        else:
            return "Lead"
    
    def fallback_prediction(self, request: PredictionRequest):
        """Fallback to rule-based prediction if ML fails"""
        # Your existing salary calculation logic
        base_salaries = {
            "Software Engineer": 800000, "Data Scientist": 900000,
            "Frontend Developer": 700000, "Backend Developer": 750000,
            "Full Stack Developer": 850000, "DevOps Engineer": 950000,
            "Product Manager": 1200000, "ML Engineer": 1000000, "Data Analyst": 600000
        }
        
        city_multipliers = {
            "Bangalore": 1.2, "Mumbai": 1.3, "Delhi": 1.25,
            "Hyderabad": 1.1, "Pune": 1.1, "Chennai": 1.0,
            "Gurgaon": 1.3, "Noida": 1.1, "Kolkata": 0.9
        }
        
        base = base_salaries.get(request.job_role, 700000)
        multiplier = city_multipliers.get(request.city, 1.0)
        experience_bonus = request.experience_years * 80000
        
        salary = int((base + experience_bonus) * multiplier)
        
        # Adjustments
        if request.education == "Masters": salary = int(salary * 1.1)
        elif request.education == "PhD": salary = int(salary * 1.2)
        
        if request.company_size == "Large": salary = int(salary * 1.15)
        elif request.company_size == "Startup": salary = int(salary * 0.9)
        
        high_value_skills = ["Machine Learning", "AWS", "Docker", "TensorFlow"]
        user_skills = [skill for skill in request.skills if skill in high_value_skills]
        if user_skills: salary = int(salary * (1 + len(user_skills) * 0.05))
        
        return salary

# Initialize the ML predictor
ml_predictor = MLSalaryPredictor()

# API Endpoints
@app.get("/")
def root():
    return {
        "message": "🇮🇳 AI Salary Predictor with ML 🤖",
        "status": "active",
        "ml_model_loaded": ml_predictor.is_ml_loaded,
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
    return {
        "status": "healthy", 
        "service": "salary-predictor",
        "ml_model_loaded": ml_predictor.is_ml_loaded,
        "version": "2.0.0"
    }

@app.get("/jobs")
def get_jobs():
    jobs = [
        "Software Engineer", "Data Scientist", "Frontend Developer",
        "Backend Developer", "Full Stack Developer", "DevOps Engineer",
        "Product Manager", "ML Engineer", "Data Analyst"
    ]
    return {"jobs": sorted(jobs)}

@app.get("/cities")
def get_cities():
    cities = [
        "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune",
        "Chennai", "Gurgaon", "Noida", "Kolkata"
    ]
    return {"cities": sorted(cities)}

@app.get("/skills")
def get_skills():
    skills = [
        "Python", "Java", "JavaScript", "SQL", "AWS",
        "Docker", "Machine Learning", "React", "Node.js",
        "TensorFlow", "PyTorch", "Data Analysis"
    ]
    return {"skills": sorted(skills)}

@app.post("/predict")
def predict_salary(request: PredictionRequest):
    # Try ML prediction first
    ml_salary = None
    if ml_predictor.is_ml_loaded:
        ml_salary = ml_predictor.predict_with_ml(request)
    
    # Use ML prediction if available, otherwise fallback
    if ml_salary is not None:
        salary = ml_salary
        model_used = "XGBoost ML Model"
        confidence = min(0.95, 0.8 + (request.experience_years * 0.015))
    else:
        salary = ml_predictor.fallback_prediction(request)
        model_used = "Rule-based (Fallback)"
        confidence = 0.85
    
    # Generate message
    message = f"As {request.job_role} with {request.experience_years} years in {request.city}"
    high_value_skills = ["Machine Learning", "AWS", "Docker", "TensorFlow"]
    user_skills = [skill for skill in request.skills if skill in high_value_skills]
    if user_skills:
        message += f". High-value skills: {', '.join(user_skills)}"
    
    return PredictionResponse(
        predicted_salary=salary,
        confidence=round(confidence, 2),
        salary_range={"min": int(salary * 0.8), "max": int(salary * 1.2)},
        message=message,
        model_used=model_used
    )

if __name__ == "__main__":
    print("🚀 AI Salary Predictor with ML Starting...")
    print("📍 http://localhost:8000")
    print("🤖 ML Model Status:", "Loaded" if ml_predictor.is_ml_loaded else "Not Loaded")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
