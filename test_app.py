import json
import os

import requests

# --- Configuration ---
# The URL where your FastAPI application is running
BASE_URL = os.getenv("APP_URL", "http://127.0.0.1:8080")
# The YouTube video you want to test
YOUTUBE_URL = "https://www.youtube.com/watch?v=ydycD3iMhvc"


def test_process_endpoint():
    """
    Sends a request to the /process endpoint and prints the response.
    """
    process_url = f"{BASE_URL}/process"
    payload = {"url": YOUTUBE_URL, "generate_summary": True}

    print(f"--- Sending POST request to {process_url} ---")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(process_url, json=payload, timeout=300)  # 5-minute timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        print(f"\n--- Response ---")
        print(f"Status Code: {response.status_code}")

        # Pretty-print the JSON response
        response_data = response.json()
        print(json.dumps(response_data, indent=2))

        # Print a summary of the results
        if response_data.get("status") == "success":
            data = response_data.get("data", {})
            print("\n--- ✅ Test Passed ---")
            print(f"Title: {data.get('title')}")
            print(f"Summary: {data.get('summary', 'Not generated.')}")
        else:
            print("\n--- ❌ Test Failed ---")
            print(f"Error Message: {response_data.get('message')}")

    except requests.exceptions.RequestException as e:
        print(f"\n--- ❌ Request Failed ---")
        print(f"An error occurred: {e}")
    except json.JSONDecodeError:
        print(f"\n--- ❌ Invalid JSON Response ---")
        print(f"Could not decode JSON from response: {response.text}")


if __name__ == "__main__":
    # First, test if the server is running
    try:
        health_check = requests.get(f"{BASE_URL}/test", timeout=5)
        if health_check.status_code == 200:
            print("✅ Server is running. Proceeding with the test...")
            test_process_endpoint()
        else:
            print(f"❌ Server returned status {health_check.status_code}. Please ensure the app is running.")
    except requests.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {BASE_URL}.")
        print("Please make sure your FastAPI application is running by executing:")
        print("uvicorn app:app --host 0.0.0.0 --port 8080")
