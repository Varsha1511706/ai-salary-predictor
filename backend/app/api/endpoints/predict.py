from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pickle
import pandas as pd
import os
import numpy as np

router = APIRouter()

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
    salary_range: Dict[str, int]
    currency: str = "INR"
    message: str = ""

class SalaryPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            # Load model
            model_path = os.path.join('ml_pipeline', 'models', 'salary_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load preprocessor
            preprocessor_path = os.path.join('ml_pipeline', 'models', 'preprocessor.pkl')
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor_data = pickle.load(f)
            
            print("✅ AI Salary Predictor loaded successfully!")
            print(f"📊 Features: {len(self.preprocessor_data['feature_columns'])}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.model = None
            self.preprocessor_data = None
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make salary prediction for Indian tech roles"""
        if self.model is None or self.preprocessor_data is None:
            raise HTTPException(status_code=500, detail="Model not loaded properly")
        
        try:
            # Prepare input data
            input_data = self.prepare_features(request)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            prediction = max(200000, min(5000000, prediction))  # Reasonable bounds
            
            # Calculate confidence based on experience (more experience = more confident)
            confidence = min(0.95, 0.7 + (request.experience_years * 0.02))
            
            # Calculate salary range (±20%)
            salary_range = {
                "min": int(prediction * 0.8),
                "max": int(prediction * 1.2)
            }
            
            # Generate insightful message
            message = self.generate_insight_message(request, prediction)
            
            return PredictionResponse(
                predicted_salary=int(prediction),
                confidence=round(confidence, 2),
                salary_range=salary_range,
                message=message
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def prepare_features(self, request: PredictionRequest) -> pd.DataFrame:
        """Prepare features for prediction matching training format"""
        # Create base dataframe
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
        
        # Apply feature engineering
        df = self.apply_feature_engineering(df)
        
        # Ensure all training features are present
        for feature in self.preprocessor_data['feature_columns']:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select and order features exactly as training
        feature_df = df[self.preprocessor_data['feature_columns']]
        
        # Scale numerical features
        numerical_columns = ['experience_years', 'experience_squared']
        for col in numerical_columns:
            if col in feature_df.columns:
                feature_df[col] = self.preprocessor_data['scaler'].transform(feature_df[[col]])
        
        return feature_df
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as during training"""
        # Experience squared
        df['experience_squared'] = df['experience_years'] ** 2
        
        # Metropolitan city indicator
        metro_cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']
        df['is_metro'] = df['city'].apply(lambda x: 1 if x in metro_cities else 0)
        
        # Encode categorical variables
        categorical_columns = ['job_role', 'city', 'experience_level', 'education', 'company_size']
        
        for col in categorical_columns:
            if col in self.preprocessor_data['label_encoders']:
                try:
                    # Handle unseen labels by using most common encoding
                    le = self.preprocessor_data['label_encoders'][col]
                    if df[col].iloc[0] in le.classes_:
                        df[f'{col}_encoded'] = le.transform(df[col])
                    else:
                        df[f'{col}_encoded'] = 0  # Default to first class
                except Exception:
                    df[f'{col}_encoded'] = 0
        
        # Add skill features
        skills_list = ['python', 'java', 'sql', 'aws', 'docker', 'machine learning', 'javascript', 'react']
        for skill in skills_list:
            df[f'skill_{skill}'] = df['skills'].apply(
                lambda x: 1 if skill in str(x).lower() else 0
            )
        
        return df
    
    def get_experience_level(self, experience_years: int) -> str:
        """Convert experience years to level"""
        if experience_years <= 2:
            return "Junior"
        elif experience_years <= 5:
            return "Mid"
        elif experience_years <= 10:
            return "Senior"
        else:
            return "Lead"
    
    def generate_insight_message(self, request: PredictionRequest, salary: int) -> str:
        """Generate insightful message about the prediction"""
        base_message = f"As a {request.job_role} with {request.experience_years} years experience in {request.city}"
        
        # Add skill-based insights
        high_value_skills = ['machine learning', 'aws', 'python', 'docker']
        user_high_value_skills = [skill for skill in request.skills if skill.lower() in high_value_skills]
        
        if user_high_value_skills:
            base_message += f". Your skills in {', '.join(user_high_value_skills)} are in high demand"
        
        # Add city-based insights
        high_salary_cities = ['Bangalore', 'Mumbai', 'Delhi']
        if request.city in high_salary_cities:
            base_message += f". Being in {request.city} commands premium salaries"
        
        # Add experience insight
        if request.experience_years >= 8:
            base_message += ". Your extensive experience is highly valued"
        
        return base_message + "."

# Initialize predictor
predictor = SalaryPredictor()

@router.post("/predict", response_model=PredictionResponse)
async def predict_salary(request: PredictionRequest):
    """Predict salary based on Indian job market data"""
    return predictor.predict(request)

@router.get("/jobs")
async def get_available_jobs():
    """Get list of available job roles"""
    jobs = [
        "Software Engineer", "Senior Software Engineer", "Data Scientist",
        "Data Analyst", "ML Engineer", "Frontend Developer",
        "Backend Developer", "Full Stack Developer", "DevOps Engineer", "Product Manager"
    ]
    return {"jobs": sorted(jobs)}

@router.get("/cities")
async def get_available_cities():
    """Get list of available cities"""
    cities = [
        "Bangalore", "Hyderabad", "Pune", "Mumbai", "Delhi",
        "Chennai", "Gurgaon", "Noida", "Kolkata"
    ]
    return {"cities": sorted(cities)}

@router.get("/skills")
async def get_popular_skills():
    """Get list of popular skills"""
    skills = [
        "Python", "Java", "JavaScript", "SQL", "AWS",
        "Docker", "Machine Learning", "React", "Node.js",
        "TensorFlow", "PyTorch", "Data Analysis", "Git"
    ]
    return {"skills": sorted(skills)}
