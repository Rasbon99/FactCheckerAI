import requests

# Base URL of your FastAPI application
base_url = "http://127.0.0.1:8003"

# Test the /results endpoint with a sample text
def test_get_results():
    text = """President-elect Donald Trump has reiterated his desire for the US to acquire Greenland and the Panama Canal, calling both critical to American national security. Asked if he would rule out using military or economic force in order to take over the autonomous Danish territory or the Canal, he responded: "No, I can't assure you on either of those two. "But I can say this, we need them for economic security," he told reporters during a wide-ranging news conference at his Mar-a-Lago estate in Florida. Both Denmark and Panama have rejected any suggestion that they would give up territory."""
    response = requests.get(f"{base_url}/results", params={"text": text})
    if response.status_code == 200:
        print("GET /results successful:", response.json())
    else:
        print(f"GET /results failed with status code {response.status_code}: {response.text}")

# Test the /clean_conversations endpoint
def test_clean_conversations():
    response = requests.post(f"{base_url}/clean_conversations")
    if response.status_code == 200:
        print("POST /clean_conversations successful.")
    else:
        print(f"POST /clean_conversations failed with status code {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_get_results()
    test_clean_conversations()
