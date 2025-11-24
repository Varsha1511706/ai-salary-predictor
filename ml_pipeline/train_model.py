import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os

class StandaloneSalaryTrainer:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_and_process_data(self, data_path):
        print("📊 Loading and processing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Original data: {df.shape}")
        
        # Clean data
        df = df.drop_duplicates().dropna()
        df = df[(df['salary_inr'] >= 200000) & (df['salary_inr'] <= 5000000)]
        print(f"Cleaned data: {df.shape}")
        
        # Feature engineering
        print("🔧 Creating features...")
        
        # Skills features
        skills_list = ['python', 'java', 'sql', 'aws', 'docker', 'machine learning', 'javascript', 'react']
        for skill in skills_list:
            df[f'skill_{skill}'] = df['skills'].apply(
                lambda x: 1 if skill in str(x).lower() else 0
            )
        
        # Encode categorical variables
        categorical_columns = ['job_role', 'city', 'experience_level', 'education', 'company_size']
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        # Additional features
        df['experience_squared'] = df['experience_years'] ** 2
        metro_cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']
        df['is_metro'] = df['city'].apply(lambda x: 1 if x in metro_cities else 0)
        
        # Prepare features
        self.feature_columns = ['experience_years', 'experience_squared', 'is_metro']
        for col in categorical_columns:
            self.feature_columns.append(f'{col}_encoded')
        
        skill_columns = [col for col in df.columns if col.startswith('skill_')]
        self.feature_columns.extend(skill_columns)
        
        X = df[self.feature_columns]
        y = df['salary_inr']
        
        print(f"🎯 Final features: {len(self.feature_columns)}")
        return X, y
    
    def train_model(self, data_path):
        print("🚀 Starting model training...")
        
        # Process data
        X, y = self.load_and_process_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale numerical features
        numerical_columns = ['experience_years', 'experience_squared']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
        X_test_scaled[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
        
        print(f"📈 Training on {X_train_scaled.shape[0]} samples")
        
        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        test_predictions = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        
        print("="*50)
        print("📊 TRAINING RESULTS:")
        print(f"💰 MAE: ₹{mae:,.0f}")
        print(f"🎯 R²: {r2:.3f}")
        print(f"💵 Avg Salary: ₹{y_test.mean():,.0f}")
        print("="*50)
        
        return mae, r2
    
    def save_model(self, model_path, preprocessor_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save preprocessor data
        with open(preprocessor_path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
        
        print(f"💾 Model saved: {model_path}")
        print(f"💾 Preprocessor saved: {preprocessor_path}")

if __name__ == "__main__":
    trainer = StandaloneSalaryTrainer()
    data_path = os.path.join('data', 'indian_salary_data.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please create data first!")
        exit(1)
    
    mae, r2 = trainer.train_model(data_path)
    trainer.save_model('models/salary_model.pkl', 'models/preprocessor.pkl')
    print("✅ Training completed!")
