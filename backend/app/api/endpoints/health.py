from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    # Check if model files exist
    model_path = os.path.join('ml_pipeline', 'models', 'salary_model.pkl')
    preprocessor_path = os.path.join('ml_pipeline', 'models', 'preprocessor.pkl')
    
    model_exists = os.path.exists(model_path)
    preprocessor_exists = os.path.exists(preprocessor_path)
    
    status = "healthy" if model_exists and preprocessor_exists else "degraded"
    
    return {
        "status": status,
        "service": "salary-predictor",
        "model_loaded": model_exists,
        "preprocessor_loaded": preprocessor_exists,
        "version": "2.0.0"
    }
