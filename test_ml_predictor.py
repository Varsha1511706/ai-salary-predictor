import requests
import json

print("🧪 TESTING ML-POWERED SALARY PREDICTOR 🤖")
print("=" * 60)

BASE_URL = "http://localhost:8000"

# Test ML predictions
test_cases = [
    {
        "name": "ML Engineer with AI Skills",
        "data": {
            "job_role": "ML Engineer",
            "experience_years": 4,
            "skills": ["Python", "Machine Learning", "TensorFlow", "AWS"],
            "city": "Bangalore",
            "education": "Masters",
            "company_size": "Large"
        }
    },
    {
        "name": "Data Scientist with Cloud Skills", 
        "data": {
            "job_role": "Data Scientist",
            "experience_years": 5,
            "skills": ["Python", "SQL", "AWS", "Docker", "Machine Learning"],
            "city": "Mumbai",
            "education": "Masters",
            "company_size": "Medium"
        }
    },
    {
        "name": "Software Engineer Standard",
        "data": {
            "job_role": "Software Engineer",
            "experience_years": 2,
            "skills": ["Java", "SQL", "JavaScript"],
            "city": "Pune",
            "education": "Bachelors",
            "company_size": "Startup"
        }
    }
]

print("🎯 TESTING ML PREDICTIONS:")
print("-" * 40)

for i, case in enumerate(test_cases, 1):
    print(f"\n📊 Case {i}: {case['name']}")
    try:
        response = requests.post(f"{BASE_URL}/predict", json=case['data'], timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ PREDICTION SUCCESSFUL!")
            print(f"   🤖 Model: {result['model_used']}")
            print(f"   💰 Salary: ₹{result['predicted_salary']:,}")
            print(f"   🎯 Confidence: {result['confidence']*100}%")
            print(f"   📊 Range: ₹{result['salary_range']['min']:,} - ₹{result['salary_range']['max']:,}")
            print(f"   💡 {result['message']}")
        else:
            print(f"   ❌ FAILED - Status {response.status_code}")
    except Exception as e:
        print(f"   ❌ ERROR - {e}")

print("\n" + "=" * 60)
print("🎉 ML-POWERED SALARY PREDICTOR IS READY! 🤖")
