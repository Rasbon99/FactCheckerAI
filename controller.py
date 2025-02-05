import requests

url = "http://127.0.0.1:8000/start"
response = requests.post(url)

url = "http://127.0.0.1:8001/process"
data = {"text": "President-elect Donald Trump has reiterated his desire for the US to acquire Greenland and the Panama Canal, calling both critical to American national security."}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())