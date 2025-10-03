import pytest
from flask import Flask
import json
from app import app, answer_from_knowledgebase, search_knowledgebase, answer_as_chatbot

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_kbanswer(client):
    response = client.post('/kbanswer', json={'message': 'What is Python?'})
    data = json.loads(response.data)
    assert response.status_code == 200
    assert 'message' in data

def test_search(client):
    response = client.post('/search', json={'message': 'Python tutorials'})
    data = json.loads(response.data)
    assert response.status_code == 200
    assert 'message' in data

def test_answer(client):
    response = client.post('/answer', json={'message': 'What is lambda function in Python?'})
    data = json.loads(response.data)
    assert response.status_code == 200
    assert 'message' in data

def test_answer_from_knowledgebase():
    message = "What is Python?"
    response_message = answer_from_knowledgebase(message)
    assert isinstance(response_message, str)

def test_search_knowledgebase():
    message = "Python tutorials"
    response_message = search_knowledgebase(message)
    assert isinstance(response_message, str)

def test_answer_as_chatbot():
    message = "What is lambda function in Python?"
    response_message = answer_as_chatbot(message)
    assert isinstance(response_message, str)