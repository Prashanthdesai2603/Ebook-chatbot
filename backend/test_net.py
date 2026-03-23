import requests
try:
    r = requests.get("https://generativelanguage.googleapis.com/")
    print("Status code:", r.status_code)
except Exception as e:
    print("Request failed:", e)
