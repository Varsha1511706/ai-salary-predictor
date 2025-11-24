import requests

print("🧪 TESTING ML PREDICTIONS...")
print("=" * 50)

test_cases = [
    {"job_role": "ML Engineer", "experience_years": 4, "skills": ["Python", "Machine Learning", "TensorFlow", "AWS"], "city": "Bangalore", "education": "Masters", "company_size": "Large"},
    {"job_role": "Data Scientist", "experience_years": 5, "skills": ["Python", "SQL", "AWS", "Docker", "Machine Learning"], "city": "Mumbai", "education": "Masters", "company_size": "Medium"},
    {"job_role": "Software Engineer", "experience_years": 2, "skills": ["Java", "SQL", "JavaScript"], "city": "Pune", "education": "Bachelors", "company_size": "Startup"}
]

for i, data in enumerate(test_cases, 1):
    response = requests.post("http://localhost:8000/predict", json=data)
    result = response.json()
    print(f"📊 Case {i}: {data['job_role']} in {data['city']}")
    print(f"   🤖 Model: {result['model_used']}")
    print(f"   💰 Salary: ₹{result['predicted_salary']:,}")
    print(f"   📊 Range: ₹{result['salary_range']['min']:,} - ₹{result['salary_range']['max']:,}")
    print()
