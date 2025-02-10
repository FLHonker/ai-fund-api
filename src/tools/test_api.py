import requests, os

# os.environ['http_proxy'] = 'socks5://127.0.0.1:1082'
# os.environ['https_proxy'] = 'socks5://127.0.0.1:1082'

BASE_URL = "https://aifund.koyeb.app"


payload = {
    "ticker": "600660",
    "start_date": "2025-01-01",
    "end_date": "2025-01-10",
    "num_of_news": 5,
    "show_reasoning": False
}
response = requests.post(f"{BASE_URL}/pred", json=payload)
print(response.json())
