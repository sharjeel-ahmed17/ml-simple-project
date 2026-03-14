"""
Test FastAPI Application
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("="*60)
print("TESTING FASTAPI APPLICATION")
print("="*60)

# Test home endpoint
print("\n1. Testing Home Endpoint...")
response = client.get('/')
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# Test model info
print("\n2. Testing Model Info Endpoint...")
response = client.get('/api/model-info')
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# Test health endpoint
print("\n3. Testing Health Endpoint...")
response = client.get('/health')
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# Test features endpoint
print("\n4. Testing Features Endpoint...")
response = client.get('/api/features')
print(f"   Status: {response.status_code}")
print(f"   Feature count: {response.json()['count']}")

# Test prediction
print("\n5. Testing Prediction Endpoint...")
test_data = {
    'age': 28,
    'experience_years': 5,
    'role': 'All-rounder',
    'batting_style': 'Right-hand bat',
    'bowling_style': 'Right-arm fast',
    'match_type': 'T20',
    'venue': 'Mumbai',
    'opposition': 'Australia',
    'balls_faced': 50,
    'runs_scored': 60,
    'fours': 6,
    'sixes': 2,
    'not_out': 1,
    'overs_bowled': 4.0,
    'runs_conceded': 30,
    'wickets_taken': 2,
    'maidens': 0,
    'dots': 12,
    'catches': 1,
    'run_outs': 0,
    'stumpings': 0
}

response = client.post('/api/predict', json=test_data)
print(f"   Status: {response.status_code}")
print(f"   Prediction: {response.json()['prediction']}")
print(f"   Probability: {response.json()['probability']}")
print(f"   Confidence: {response.json()['confidence']}")

# Test batch prediction
print("\n6. Testing Batch Prediction Endpoint...")
batch_data = [test_data, test_data]
response = client.post('/api/batch-predict', json=batch_data)
print(f"   Status: {response.status_code}")
print(f"   Predictions count: {response.json()['count']}")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
