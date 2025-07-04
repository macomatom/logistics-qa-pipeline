import requests

url = "https://gotify.martinbaca.synology.me/message"
token = "A682soruZxn_aTW"

def send_gotify_notification(message):
    url = "https://gotify.martinbaca.synology.me/message"
    token = "A682soruZxn_aTW"
    data = {"message": message, "priority": 5, "title": "Jupyter Notebook"}
    headers = {"X-Gotify-Key": token}

    requests.post(url, json=data, headers=headers)
    
def get_gotify_url():
    return url

def get_gotify_token():
    return token