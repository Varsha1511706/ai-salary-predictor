import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "AI Salary Predictor"
    PROJECT_VERSION: str = "1.0.0"
    
    # Model paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "ml_pipeline/models/salary_model.pkl")
    PREPROCESSOR_PATH: str = os.getenv("PREPROCESSOR_PATH", "ml_pipeline/models/preprocessor.pkl")
    
    # API settings
    API_V1_STR: str = "/api/v1"

settings = Settings()