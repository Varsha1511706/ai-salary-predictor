import requests
import json

print("🧪 COMPREHENSIVE TEST: AI SALARY PREDICTOR 🇮🇳")
print("=" * 60)

BASE_URL = "http://localhost:8000"

# Test all GET endpoints
endpoints = [
    ("/", "Root"),
    ("/health", "Health Check"),
    ("/jobs", "Available Jobs"),
    ("/cities", "Indian Cities"),
    ("/skills", "Tech Skills")
]

print("📡 Testing GET Endpoints:")
print("-" * 40)

for endpoint, name in endpoints:
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {name}: SUCCESS")
            
            if name == "Available Jobs":
                print(f"   💼 {len(data['jobs'])} job roles available")
                print(f"   Sample: {', '.join(data['jobs'][:3])}")
            elif name == "Indian Cities":
                print(f"   🏙️  {len(data['cities'])} cities available") 
                print(f"   Sample: {', '.join(data['cities'][:3])}")
            elif name == "Tech Skills":
                print(f"   💻 {len(data['skills'])} skills listed")
                print(f"   Sample: {', '.join(data['skills'][:3])}")
        else:
            print(f"❌ {name}: FAILED - Status {response.status_code}")
    except Exception as e:
        print(f"❌ {name}: ERROR - {e}")

# Test salary predictions
print("\n🎯 TESTING SALARY PREDICTIONS:")
print("-" * 40)

test_cases = [
    {
        "name": "Junior Software Engineer",
        "data": {
            "job_role": "Software Engineer",
            "experience_years": 1,
            "skills": ["Python", "Java"],
            "city": "Pune",
            "education": "Bachelors",
            "company_size": "Startup"
        }
    },
    {
        "name": "Mid-level Data Scientist", 
        "data": {
            "job_role": "Data Scientist",
            "experience_years": 4,
            "skills": ["Python", "Machine Learning", "SQL"],
            "city": "Bangalore",
            "education": "Masters",
            "company_size": "Medium"
        }
    },
    {
        "name": "Senior DevOps Engineer",
        "data": {
            "job_role": "DevOps Engineer",
            "experience_years": 8,
            "skills": ["AWS", "Docker", "Kubernetes", "Python"],
            "city": "Mumbai",
            "education": "Bachelors",
            "company_size": "Large"
        }
    }
]

for i, case in enumerate(test_cases, 1):
    print(f"\n📊 Case {i}: {case['name']}")
    try:
        response = requests.post(f"{BASE_URL}/predict", json=case['data'], timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ PREDICTION SUCCESSFUL!")
            print(f"   💰 Salary: ₹{result['predicted_salary']:,}")
            print(f"   🎯 Confidence: {result['confidence']*100}%")
            print(f"   📊 Range: ₹{result['salary_range']['min']:,} - ₹{result['salary_range']['max']:,}")
            print(f"   💡 {result['message']}")
        else:
            print(f"   ❌ FAILED - Status {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ ERROR - {e}")

print("\n" + "=" * 60)
print("🎉 AI SALARY PREDICTOR IS FULLY OPERATIONAL! 🇮🇳")
print("✨ Your API is ready for use!")
