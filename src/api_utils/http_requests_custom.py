import requests

def make_post_request(prompt, model, api_url, timeout=60):
    """Sends a POST request to the API with the given prompt.

    Args:
        prompt (str): The input prompt for the API.
        model (str): The model name to use.
        api_url (str): The API endpoint URL.
        timeout (int, optional): Timeout in seconds. Defaults to 60.

    Returns:
        requests.Response or None: The API response object, or None on failure.
    """
    payload = {"model": model, "prompt": prompt}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None