#!/usr/bin/env python3
"""
Script to test the chatbot API endpoint
"""
import requests
import json

def test_chatbot():
    # Test the chatbot API endpoint
    api_url = "http://127.0.0.1:8000/api/chat"

    # Headers (using the default API key from the code)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "63f4945d921d599f27ae4fdf5bada3f1"  # Default API key from code
    }

    # Test questions
    test_questions = [
        "What is physical AI?",
        "Explain humanoid robotics",
        "How does the RAG system work?",
        "What are the main modules in this book?"
    ]

    print("Testing chatbot API endpoint...\n")

    for i, question in enumerate(test_questions, 1):
        print(f"Test {i}: Question: '{question}'")

        # Prepare the request payload
        payload = {
            "question": question,
            "context": None,
            "session_id": None
        }

        try:
            # Send the request
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30  # 30 second timeout
            )

            print(f"  Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"  Answer: {result.get('answer', 'No answer field')[:200]}...")
                print(f"  Sources: {len(result.get('sources', []))} source(s)")
                print(f"  Metadata: {result.get('metadata', {})}")
            else:
                print(f"  Error: {response.text[:200]}...")

        except requests.exceptions.ConnectionError:
            print("  Error: Could not connect to the API server. Make sure it's running on http://127.0.0.1:8000")
            break
        except requests.exceptions.Timeout:
            print("  Error: Request timed out after 30 seconds")
        except Exception as e:
            print(f"  Error: {str(e)}")

        print()  # Empty line for readability

if __name__ == "__main__":
    test_chatbot()