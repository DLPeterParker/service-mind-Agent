import requests

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3VzZXJfOTk5In0.inocgJ6Fqso9j2N-KbnETvxOWwdXU_XcFymGF4u27wo"
URL = "http://127.0.0.1:8000/chat"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

body = {
    "user_id": "test_user_999",
    "message": "你好",
}

response = requests.post(URL, headers=headers, json=body)
print(f"状态码: {response.status_code}")
print(f"响应: {response.text}")
