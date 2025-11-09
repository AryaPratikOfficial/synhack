# Requires: pip install faker pandas
from faker import Faker
import random
import pandas as pd

fake = Faker()
Faker.seed(42)
random.seed(42)

cities = ["New York","San Francisco","Chicago","Austin","Seattle","Boston","Los Angeles","Atlanta","Denver","Miami","San Diego","Portland","Philadelphia","Phoenix","Dallas","Houston"]
# City coordinates (latitude, longitude)
city_coords = {
    "New York": (40.7128, -74.0060),
    "San Francisco": (37.7749, -122.4194),
    "Chicago": (41.8781, -87.6298),
    "Austin": (30.2672, -97.7431),
    "Seattle": (47.6062, -122.3321),
    "Boston": (42.3601, -71.0589),
    "Los Angeles": (34.0522, -118.2437),
    "Atlanta": (33.7490, -84.3880),
    "Denver": (39.7392, -104.9903),
    "Miami": (25.7617, -80.1918),
    "San Diego": (32.7157, -117.1611),
    "Portland": (45.5152, -122.6784),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "Dallas": (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698)
}
roles = ["finance","operations","marketing","technology","design","production","generalist","human Resources"]
skill_pools = {
    "technology": ["Python","Java","C#","JavaScript","SQL","AWS","Docker","Kubernetes","React","Node.js","Go"],
    "finance": ["Financial Modeling","Excel","Accounting","Budgeting","Variance Analysis","Taxation","ERP"],
    "operations": ["Supply Chain","Operations Management","ERP","Lean","Process Improvement","Excel"],
    "marketing": ["SEO","Content Writing","Social Media","Google Ads","Analytics","Photoshop","Copywriting"],
    "design": ["Photoshop","Illustrator","Figma","Sketch","Adobe XD","UI/UX"],
    "production": ["Production Planning","Quality Control","Lean Manufacturing","Maintenance","SOPs"],
    "generalist": ["Customer Service","Salesforce","Excel","Reporting","Coordination"],
    "human Resources": ["Recruiting","Onboarding","HR Policies","Performance Management","LMS"]
}

def sample_skills(role):
    pool = skill_pools.get(role, sum(skill_pools.values(), []))
    count = random.choices([1,2,3,4,5], weights=[10,25,35,20,10])[0]
    return ", ".join(random.sample(pool, min(count, len(pool))))

def sample_ctc(exp):
    # base by role seniority and randomness; return value in thousands
    base = 30 + exp * random.uniform(4.0,9.0)
    noise = random.gauss(0,15)
    return max(20, round(base + noise))

rows = []
N = 200
for i in range(1, N+1):
    role = random.choice(roles)
    city = random.choice(cities)
    # skew experience by role typicals
    if role == "production":
        exp = max(0, int(random.gauss(8,4)))
    elif role == "technology":
        exp = max(0, int(random.gauss(5,3)))
    elif role == "finance":
        exp = max(0, int(random.gauss(6,3)))
    elif role == "marketing":
        exp = max(0, int(random.gauss(4,3)))
    elif role == "human Resources":
        exp = max(0, int(random.gauss(5,3)))
    else:
        exp = max(0, int(random.gauss(4,3)))

    skills = sample_skills(role)
    ctc = sample_ctc(exp)
    lat, lon = city_coords.get(city, (0.0, 0.0))
    github_username = fake.user_name().lower()
    rows.append({
        "id": f"C{i:04d}",
        "name": fake.name(),
        "city": city,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "experience_years": int(exp),
        "skills": skills,
        "ctc_expectation_k": int(ctc),
        "role_applying": role,
        "github_url": f"https://github.com/{github_username}"
    })

df = pd.DataFrame(rows)
df.to_csv("candidates.csv", index=False)
df.to_json("candidates.json", orient="records", lines=False)
print("Generated candidates.csv and candidates.json (N=%d)"%N)
# Pretty JSON (array of records)
records = df.to_dict(orient="records")
with open("candidates_pretty.json", "w", encoding="utf-8") as f:
    import json
    json.dump(records, f, ensure_ascii=False, indent=2)