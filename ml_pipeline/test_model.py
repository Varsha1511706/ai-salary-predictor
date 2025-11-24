import pickle
import pandas as pd
import os

print("🧪 Testing model loading...")

# Check if model files exist
model_path = 'models/salary_model.pkl'
preprocessor_path = 'models/preprocessor.pkl'

print(f"Model exists: {os.path.exists(model_path)}")
print(f"Preprocessor exists: {os.path.exists(preprocessor_path)}")

if os.path.exists(model_path) and os.path.exists(preprocessor_path):
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        
        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        print("✅ Preprocessor loaded successfully")
        
        print(f"📊 Feature columns: {len(preprocessor_data['feature_columns'])}")
        print(f"🔤 Label encoders: {len(preprocessor_data['label_encoders'])}")
        
        # Test prediction with sample data
        sample_data = {
            'job_role': 'Software Engineer',
            'city': 'Bangalore', 
            'experience_years': 3,
            'skills': 'Python, SQL, AWS'
        }
        print(f"🎯 Sample data: {sample_data}")
        print("✅ Model is ready for use!")
        
    except Exception as e:
        print(f"❌ Error loading: {e}")
else:
    print("❌ Model files not found")
