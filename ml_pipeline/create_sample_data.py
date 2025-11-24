import pandas as pd
import random

# Create sample data
jobs = ['Software Engineer', 'Data Scientist', 'Frontend Developer', 'Backend Developer']
cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune']
skills_list = ['Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'React']

data = []
for i in range(200):  # Create 200 samples
    job = random.choice(jobs)
    city = random.choice(cities)
    exp = random.randint(1, 10)
    
    # Base salary calculation
    if job == 'Software Engineer':
        base = 800000
    elif job == 'Data Scientist':
        base = 900000
    elif job == 'Frontend Developer':
        base = 700000
    else:
        base = 750000
    
    # Adjust for experience
    salary = base + (exp * 80000)
    
    # Adjust for city
    if city == 'Bangalore':
        salary *= 1.2
    elif city == 'Mumbai':
        salary *= 1.3
    elif city == 'Delhi':
        salary *= 1.25
    
    # Add random variation
    salary = salary * random.uniform(0.9, 1.1)
    
    # Select skills
    num_skills = random.randint(2, 4)
    skills = random.sample(skills_list, num_skills)
    
    data.append({
        'job_role': job,
        'city': city,
        'experience_years': exp,
        'experience_level': 'Junior' if exp <= 3 else 'Senior' if exp <= 6 else 'Lead',
        'skills': ', '.join(skills),
        'salary_inr': int(salary),
        'education': random.choice(['Bachelors', 'Masters']),
        'company_size': random.choice(['Startup', 'Medium', 'Large'])
    })

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('data/indian_salary_data.csv', index=False)
print(f"Created {len(df)} sample records")
print("First 3 records:")
print(df.head(3))
