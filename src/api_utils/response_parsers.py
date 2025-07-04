import json

def extract_json_responses(response):
    """Extracts and concatenates JSON responses from a multi-line API response.
    
    Args:
        response (requests.Response): The API response object.

    Returns:
        str: Extracted content as a single string.
    """
    try:
        content_array = response.content.decode().strip().split("\n")
        content_array = [json.loads(j)["response"] for j in content_array if j.strip()]
        return "".join(content_array)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON response: {e}")
        return ""