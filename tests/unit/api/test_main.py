#!/usr/bin/env python
# tests/unit/api/test_main.py
# Unit tests for API endpoints

import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the FastAPI app
from api.main import app

# Create test client
client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    
    # Check status code
    assert response.status_code == 200
    
    # Check response content
    data = response.json()
    assert "name" in data
    assert data["name"] == "LinguaSign API"
    assert "version" in data

@patch("api.main.TranslationResponse")
def test_translate(mock_translation_response):
    """Test the translation endpoint."""
    # Mock the response
    mock_instance = MagicMock()
    mock_instance.text = "Hello world"
    mock_instance.confidence = 0.95
    mock_translation_response.return_value = mock_instance
    
    # Create a test file
    with open("test_video.mp4", "wb") as f:
        f.write(b"test video content")
    
    # Make the request
    with open("test_video.mp4", "rb") as f:
        response = client.post(
            "/api/v1/translate",
            files={"file": ("test_video.mp4", f, "video/mp4")}
        )
    
    # Clean up the test file
    os.remove("test_video.mp4")
    
    # Check status code
    assert response.status_code == 200
    
    # Check response content
    data = response.json()
    assert "text" in data
    assert data["text"] == "Hello world"
    assert "confidence" in data
    assert data["confidence"] == 0.95

def test_get_signs():
    """Test the get signs endpoint."""
    response = client.get("/api/v1/signs")
    
    # Check status code
    assert response.status_code == 200
    
    # Check response content
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Check the structure of the first item
    first_item = data[0]
    assert "id" in first_item
    assert "name" in first_item
    assert "description" in first_item

def test_missing_file():
    """Test translation endpoint with missing file."""
    # Make the request without a file
    response = client.post("/api/v1/translate")
    
    # Check status code
    assert response.status_code == 422  # Unprocessable Entity
