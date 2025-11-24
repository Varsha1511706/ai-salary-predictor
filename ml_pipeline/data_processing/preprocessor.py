import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class SalaryDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, file_path):
        print(f"Loading data from: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        return pd.read_csv(file_path)
    
    def clean_data(self, df):
        print(f"Original data shape: {df.shape}")
        df = df.drop_duplicates()
        df = df.dropna()
        df = df[(df['salary_inr'] >= 200000) & (df['salary_inr'] <= 5000000)]
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        print("Creating features...")
        
        # Create skill features
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
        
        # Create additional features
        df['experience_squared'] = df['experience_years'] ** 2
        
        # Metro city indicator
        metro_cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']
        df['is_metro'] = df['city'].apply(lambda x: 1 if x in metro_cities else 0)
        
        return df
    
    def prepare_features(self, df):
        feature_columns = ['experience_years', 'experience_squared', 'is_metro']
        
        # Add encoded categorical features
        categorical_columns = ['job_role', 'city', 'experience_level', 'education', 'company_size']
        for col in categorical_columns:
            feature_columns.append(f'{col}_encoded')
        
        # Add skill features
        skill_columns = [col for col in df.columns if col.startswith('skill_')]
        feature_columns.extend(skill_columns)
        
        self.feature_columns = feature_columns
        
        X = df[feature_columns]
        y = df['salary_inr']
        
        print(f"Final feature count: {len(feature_columns)}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def scale_features(self, X_train, X_test):
        numerical_columns = ['experience_years', 'experience_squared']
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
        X_test_scaled[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessor(self, file_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
        print(f"Preprocessor saved to {file_path}")

if __name__ == "__main__":
    preprocessor = SalaryDataPreprocessor()
    
    # Use absolute path to be sure
    data_path = os.path.join(os.getcwd(), 'data', 'indian_salary_data.csv')
    print(f"Looking for data at: {data_path}")
    
    df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(df)
    df = preprocessor.feature_engineering(df)
    
    X, y = preprocessor.prepare_features(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Create models directory
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    preprocessor.save_preprocessor(os.path.join(models_dir, 'preprocessor.pkl'))
    
    print(f"Data processing completed!")
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Features: {len(preprocessor.feature_columns)}")
    print(f"Salary range: ₹{y.min():,} to ₹{y.max():,}")
