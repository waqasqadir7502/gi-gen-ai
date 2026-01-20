#!/usr/bin/env python3
"""
Test script to verify the chat API is working properly
"""
import requests
import json

def test_chat_api():
    print("Testing the chat API endpoint...")

    # Test the health endpoint first
    try:
        health_response = requests.get("http://127.0.0.1:8000/health")
        print(f"Health check status: {health_response.status_code}")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Health data: {health_data}")
        else:
            print(f"Health check failed: {health_response.text}")
    except Exception as e:
        print(f"Error checking health: {e}")
        return

    # Test the chat endpoint
    try:
        headers = {
            "Content-Type": "application/json",
        }

        # Test question
        payload = {
            "question": "What is physical AI?",
            "context": None,
            "session_id": "test-session-123"
        }

        response = requests.post(
            "http://127.0.0.1:8000/api/chat",
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"Chat API status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Answer received: {result.get('answer', 'No answer field')[:200]}...")
            print(f"Sources: {len(result.get('sources', []))} sources")
            print("✓ Chat API is working correctly!")
        else:
            print(f"Chat API error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to the API server. Make sure it's running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"✗ Error testing chat API: {e}")

if __name__ == "__main__":
    test_chat_api()