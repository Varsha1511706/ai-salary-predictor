import pandas as pd
import numpy as np
import random
from datetime import datetime

class IndianSalaryDataGenerator:
    def __init__(self):
        self.sample_data = []
        
    def generate_sample_data(self, num_samples=1000):
        # Indian job roles with typical salary ranges (in INR LPA)
        job_roles = {
            'Software Engineer': {'min': 6, 'max': 18},
            'Senior Software Engineer': {'min': 12, 'max': 25},
            'Data Scientist': {'min': 8, 'max': 20},
            'Data Analyst': {'min': 5, 'max': 15},
            'ML Engineer': {'min': 9, 'max': 22},
            'Frontend Developer': {'min': 5, 'max': 16},
            'Backend Developer': {'min': 6, 'max': 18},
            'Full Stack Developer': {'min': 7, 'max': 19},
            'DevOps Engineer': {'min': 8, 'max': 20},
            'Product Manager': {'min': 12, 'max': 30}
        }
        
        # Indian cities with cost of living multipliers
        cities = {
            'Bangalore': 1.2, 'Hyderabad': 1.0, 'Pune': 1.0,
            'Mumbai': 1.4, 'Delhi': 1.3, 'Chennai': 0.9,
            'Gurgaon': 1.3, 'Noida': 1.1, 'Kolkata': 0.8
        }
        
        # Skills mapping
        skills_mapping = {
            'Software Engineer': ['Python', 'Java', 'SQL', 'AWS', 'Docker'],
            'Data Scientist': ['Python', 'Machine Learning', 'SQL', 'Statistics'],
            'Data Analyst': ['SQL', 'Excel', 'Python', 'Tableau'],
            'ML Engineer': ['Python', 'TensorFlow', 'PyTorch', 'AWS'],
            'Frontend Developer': ['JavaScript', 'React', 'HTML', 'CSS'],
            'Backend Developer': ['Python', 'Java', 'Node.js', 'SQL'],
            'Full Stack Developer': ['JavaScript', 'React', 'Node.js', 'Python'],
            'DevOps Engineer': ['AWS', 'Docker', 'Kubernetes', 'Jenkins'],
            'Product Manager': ['Product Strategy', 'SQL', 'Agile', 'JIRA']
        }
        
        for i in range(num_samples):
            role = random.choice(list(job_roles.keys()))
            city = random.choice(list(cities.keys()))
            experience = random.randint(1, 15)
            education = random.choice(['Bachelors', 'Masters', 'PhD'])
            company_size = random.choice(['Startup', 'Medium', 'Large'])
            
            # Calculate base salary
            role_info = job_roles[role]
            base_salary = random.uniform(role_info['min'], role_info['max'])
            
            # Adjust for experience
            if experience <= 2:
                level = 'Junior'
                salary = base_salary * 0.6
            elif experience <= 5:
                level = 'Mid'
                salary = base_salary * 0.8
            elif experience <= 10:
                level = 'Senior'
                salary = base_salary * 1.0
            else:
                level = 'Lead'
                salary = base_salary * 1.3
            
            # Adjust for city
            salary = salary * cities[city]
            
            # Adjust for education
            if education == 'Masters':
                salary *= 1.1
            elif education == 'PhD':
                salary *= 1.2
            
            # Adjust for company size
            if company_size == 'Large':
                salary *= 1.15
            elif company_size == 'Startup':
                salary *= 0.9
            
            # Add random variation
            salary = salary * random.uniform(0.9, 1.1)
            
            # Convert to annual salary in INR
            annual_salary = int(salary * 100000)
            
            # Select skills
            available_skills = skills_mapping.get(role, ['Python', 'SQL'])
            num_skills = random.randint(3, min(5, len(available_skills)))
            skills = random.sample(available_skills, num_skills)
            
            self.sample_data.append({
                'job_role': role,
                'city': city,
                'experience_years': experience,
                'experience_level': level,
                'skills': ', '.join(skills),
                'salary_inr': annual_salary,
                'education': education,
                'company_size': company_size
            })
        
        return pd.DataFrame(self.sample_data)
    
    def save_data(self, filename='indian_salary_data.csv'):
        df = self.generate_sample_data(500)  # Generate 500 samples for quick testing
        df.to_csv(f'data/{filename}', index=False)
        print(f"Generated {len(df)} salary records")
        print(f"Salary range: ₹{df['salary_inr'].min():,} to ₹{df['salary_inr'].max():,}")
        return df

if __name__ == "__main__":
    generator = IndianSalaryDataGenerator()
    df = generator.save_data()
    print("First 3 records:")
    print(df.head(3))
