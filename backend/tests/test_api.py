import requests
import json

# Test the API
def test_prediction():
    url = "http://localhost:8000/api/v1/predict"
    
    data = {
        "job_role": "Software Engineer",
        "experience_years": 3,
        "skills": ["Python", "SQL", "JavaScript"],
        "city": "Bangalore",
        "education": "Bachelors",
        "company_size": "Medium",
        "job_type": "Full-time"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_prediction()