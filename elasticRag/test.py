import pytest
from elasticRag_app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        return client

def test_e2e(client):
    response = client.post('/answer', json={'query': 'How many personal leaves do I have?'})
    assert response.status_code == 200
    jsonObject = response.get_json()
    assert "sources" in jsonObject
    assert "response" in jsonObject
    print("Test Successfull")

def test_retrieval(client):
    response = client.post('/retrieve', json={'query': 'How many paid leaves do I have?'})
    assert response.status_code == 200
    assert len(response.get_json()['sources']) == 5
    print("Test Successfull")
    
@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        print("All tests passed successfully!")
    else:
        print("Some tests failed.")
