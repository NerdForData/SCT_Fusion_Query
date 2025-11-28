"""
Helper to obtain GPT4IFX bearer token.

Call get_bearer_token() from other scripts.
"""

import base64
import requests
import urllib3
from typing import Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure these to your environment
BASE_URL = "https://gpt4ifx.icp.infineon.com"
USERNAME = "INFINEON\\Aishwarya"
PASSWORD = "Gauri@123456789"

def get_bearer_token(username: str = USERNAME, password: str = PASSWORD, timeout: int = 30) -> Optional[str]:
    """
    Request a Bearer token from GPT4IFX using Basic Auth.
    Returns token string or None on failure.
    """
    print("[base_file] Requesting GPT4IFX bearer token...")
    basic_token = base64.b64encode(f"{username}:{password}".encode("ascii")).decode("ascii")
    headers = {"Authorization": f"Basic {basic_token}", "Content-Type": "application/json"}

    try:
        resp = requests.get(f"{BASE_URL}/auth/token", headers=headers, verify=False, auth=(username, password), timeout=timeout)
    except Exception as e:
        print("[base_file] Token request exception:", e)
        return None

    if resp.status_code != 200:
        print(f"[base_file] Token request failed (HTTP {resp.status_code}) - response snippet: {resp.text[:300]}")
        return None

    token = None
    # Try JSON first
    try:
        data = resp.json()
        token = data.get("access_token") or data.get("token") or data.get("id_token")
    except ValueError:
        pass

    # Fallback to headers
    if not token:
        token = (resp.headers.get("x-forwarded-access-token")
                 or resp.headers.get("access-token")
                 or resp.headers.get("authorization"))

    if token and isinstance(token, str) and token.startswith("Bearer "):
        token = token[len("Bearer "):]

    if token:
        print("[base_file] Bearer token retrieved (length:", len(token), ")")
        return token

    print("[base_file] Token not found in response.")
    return None


# Test execution when run directly
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPT4IFX Bearer Token Test")
    print("=" * 60)
    
    token = get_bearer_token()
    
    if token:
        print(f"\n[SUCCESS] Token obtained successfully!")
        print(f"[INFO] Token length: {len(token)} characters")
        print(f"[INFO] Token preview: {token[:50]}...")
        print("\n" + "=" * 60)
    else:
        print("\n[ERROR] Failed to obtain token.")
        print("=" * 60)