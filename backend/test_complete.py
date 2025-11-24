import requests
import json

BASE_URL = "http://localhost:8000"

def test_all_endpoints():
    print("🧪 COMPREHENSIVE API TEST")
    print("=" * 60)
    
    endpoints = [
        ("/", "Root"),
        ("/api/v1/health", "Health Check"),
        ("/api/v1/jobs", "Available Jobs"), 
        ("/api/v1/cities", "Available Cities"),
        ("/api/v1/skills", "Popular Skills")
    ]
    
    for endpoint, description in endpoints:
        print(f"\\n📡 Testing {description}...")
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ SUCCESS - Status: {response.status_code}")
                if endpoint == "/":
                    print(f"   📝 {data['message']}")
                elif "jobs" in endpoint:
                    print(f"   💼 {len(data['jobs'])} jobs available")
                elif "cities" in endpoint:
                    print(f"   🏙️  {len(data['cities'])} cities available")
                elif "skills" in endpoint:
                    print(f"   💻 {len(data['skills'])} skills listed")
                else:
                    print(f"   📊 {data}")
            else:
                print(f"   ❌ FAILED - Status: {response.status_code}")
                print(f"   📄 Response: {response.text}")
        except Exception as e:
            print(f"   💥 ERROR: {e}")
    
    # Test prediction
    print(f"\\n🎯 Testing Salary Prediction...")
    test_data = {
        "job_role": "Software Engineer",
        "experience_years": 3,
        "skills": ["Python", "SQL", "AWS"],
        "city": "Bangalore",
        "education": "Bachelors",
        "company_size": "Medium"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print("   ✅ PREDICTION SUCCESSFUL!")
            print(f"   💰 Salary: ₹{result['predicted_salary']:,}")
            print(f"   🎯 Confidence: {result['confidence']*100}%")
            print(f"   📊 Range: ₹{result['salary_range']['min']:,} - ₹{result['salary_range']['max']:,}")
            print(f"   💡 Insight: {result['message']}")
        else:
            print(f"   ❌ PREDICTION FAILED - Status: {response.status_code}")
            print(f"   📄 Response: {response.text}")
    except Exception as e:
        print(f"   💥 PREDICTION ERROR: {e}")
    
    print("\\n" + "=" * 60)
    print("🎉 TESTING COMPLETE!")

if __name__ == "__main__":
    test_all_endpoints()
