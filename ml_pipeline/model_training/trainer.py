import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import pickle
import os
from data_processing.preprocessor import SalaryDataPreprocessor

class SalaryPredictorTrainer:
    def __init__(self):
        self.model = None
        self.preprocessor = SalaryDataPreprocessor()
        
    def train_model(self, data_path):
        print("Starting model training...")
        
        # Load and preprocess data
        df = self.preprocessor.load_data(data_path)
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.feature_engineering(df)
        
        # Prepare features
        X, y = self.preprocessor.prepare_features(df)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        print(f"Training on {X_train_scaled.shape[0]} samples with {X_train_scaled.shape[1]} features")
        
        # Train simple model first
        self.model = xgb.XGBRegressor(
            n_estimators=50,  # Reduced for quick training
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        test_predictions = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        mae_percentage = (mae / y_test.mean()) * 100
        
        print("="*50)
        print("MODEL TRAINING RESULTS:")
        print(f"Mean Absolute Error: ₹{mae:,.0f}")
        print(f"MAE as % of average: {mae_percentage:.1f}%")
        print(f"R² Score: {r2:.3f}")
        print(f"Average Salary: ₹{y_test.mean():,.0f}")
        print("="*50)
        
        return {'mae': mae, 'r2': r2}
    
    def save_model(self, model_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    trainer = SalaryPredictorTrainer()
    data_path = os.path.join(os.getcwd(), 'data', 'indian_salary_data.csv')
    metrics = trainer.train_model(data_path)
    trainer.save_model('models/salary_model.pkl')
    print("Model training completed successfully!")
# Create a simple test script to verify the model works
$test_script = @"
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
"@

$test_script | Out-File -FilePath "test_model.py" -Encoding utf8

# Run the test
python test_model.py