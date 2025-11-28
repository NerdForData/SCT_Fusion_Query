"""
GPT Prompt Set - GPT Client with GPT4IFX Integration
-----------------------------------------------------

Purpose:
    Python client wrapper for GPT-4o API with support for GPT4IFX
    (Infineon internal endpoint). Manages authentication, rules-based
    system prompts, and response word count enforcement.

Features:
    - GPT4IFX authentication with Bearer token generation
    - Rules-based system prompts loaded from external file
    - Word count enforcement (200-word limit with auto-retry)
    - Retry logic with exponential backoff for API failures
    - SSL verification disabled for internal Infineon certificates

Configuration:
    - Base URL: https://gpt4ifx.icp.infineon.com
    - Default Model: gpt-4o (configurable via GPT_MODEL env var)
    - Default Temperature: 0.4 (configurable via GPT_TEMPERATURE env var)
    - Max Retries: 2 attempts for word count compliance
    - Timeout: 10 seconds per API request

Usage:
    from gpt_client import GPTClient
    
    client = GPTClient(rules_path="rules.txt", model="gpt-4o", temperature=0.4)
    answer = client.ask("What are the temperature specifications?")
    print(answer)

Authentication:
    Uses Basic Auth to retrieve Bearer token from GPT4IFX endpoint.
    Token is stored in OpenAI client for subsequent requests.
"""

import os
import time
import base64
import requests
from typing import List, Dict, Optional
import urllib3

# Disable SSL warnings for internal Infineon network
# GPT4IFX uses self-signed certificates that trigger warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------
# OPENAI SDK IMPORTS
# ---------------------------------------------------------
# Import OpenAI SDK for GPT API interaction
try:
    from openai import OpenAI
    import httpx
except Exception:
    OpenAI = None

# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def _load_rules(path: str = "rules.txt") -> str:
    """
    Load system prompt rules from external text file.
    
    Rules file contains domain-specific context, constraints, and
    instructions that are injected as system message in every query.
    
    Args:
        path (str): Relative path to rules file (default: "rules.txt")
    
    Returns:
        str: Content of rules file (stripped of leading/trailing whitespace)
    """
    # Load rules relative to this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rules_path = os.path.join(base_dir, path)
    with open(rules_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Default settings (overridable via environment variables)
DEFAULT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", "0.4"))  # Balanced creativity
DEFAULT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")                 # GPT-4o model
MAX_RETRIES = 3           # Maximum retry attempts for API failures
RETRY_BACKOFF = 2.0       # Exponential backoff multiplier

# ---------------------------------------------------------
# GPT CLIENT CLASS
# ---------------------------------------------------------

class GPTClient:
    """
    GPT Client wrapper for GPT4IFX API interaction.
    
    Manages authentication, rules loading, and query execution with
    word count enforcement and retry logic.
    
    Attributes:
        rules (str): System prompt rules loaded from file
        model (str): GPT model identifier (e.g., "gpt-4o")
        temperature (float): Sampling temperature (0.0-1.0)
        client (OpenAI): Authenticated OpenAI client instance
    """
    
    def __init__(self,
                 rules_path: str = "rules.txt",
                 model: str = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE):
        """
        Initialize GPT client with rules and configuration.
        
        Args:
            rules_path (str): Path to rules file (default: "rules.txt")
            model (str): GPT model to use (default: from env or "gpt-4o")
            temperature (float): Sampling temperature (default: from env or 0.4)
        
        Raises:
            ImportError: If OpenAI SDK is not installed
            FileNotFoundError: If rules file doesn't exist
        """
        # Validate OpenAI SDK availability
        if OpenAI is None:
            raise ImportError("openai SDK not available. Install it with pip install openai.")
        
        # Load system prompt rules from file
        self.rules = _load_rules(rules_path)
        self.model = model
        self.temperature = temperature
        
        # Setup GPT4IFX authentication and client
        self._setup_gpt4ifx_client()

    def _setup_gpt4ifx_client(self):
        """
        Setup authenticated OpenAI client for GPT4IFX endpoint.
        
        Authentication Flow:
            1. Encode credentials as Base64 for Basic Auth
            2. Request Bearer token from /auth/token endpoint
            3. Extract token from response headers
            4. Initialize OpenAI client with Bearer token
        
        Falls back to standard OpenAI client if authentication fails
        (though this will likely fail without valid API key).
        """
        # ---------------------------------------------------------
        # GPT4IFX CONFIGURATION
        # ---------------------------------------------------------
        base_url = "https://gpt4ifx.icp.infineon.com"  # Internal Infineon endpoint
        username = "INFINEON\\Aishwarya"
        password = "Gauri@123456789"
        
        # ---------------------------------------------------------
        # BEARER TOKEN AUTHENTICATION
        # ---------------------------------------------------------
        # Encode credentials as Base64 for Basic Auth
        basic_token = base64.b64encode(f"{username}:{password}".encode("ascii")).decode("ascii")
        headers = {
            "Authorization": f"Basic {basic_token}",
            "Content-Type": "application/json"
        }
        
        try:
            # Request Bearer token from authentication endpoint
            resp = requests.get(
                f"{base_url}/auth/token",
                headers=headers,
                verify=False,  # Disable SSL verification for self-signed cert
                auth=(username, password)
            )
            
            # Check authentication success
            if resp.status_code == 200:
                # Extract Bearer token from response headers
                token = resp.headers.get("x-forwarded-access-token")
                if token:
                    # Setup OpenAI client with GPT4IFX endpoint and Bearer token
                    auth_headers = {"Authorization": f"Bearer {token}"}
                    self.client = OpenAI(
                        api_key=token,
                        base_url=base_url,
                        default_headers=auth_headers,
                        http_client=httpx.Client(verify=False)  # Disable SSL for internal network
                    )
                    # Model remains as specified in constructor (e.g., gpt-4o)
                    return
            
            # Authentication failed with non-200 status
            raise Exception(f"Authentication failed with status {resp.status_code}")
            
        except Exception as e:
            # Log error and fallback to standard OpenAI (will likely fail)
            print(f"Failed to setup GPT4IFX client: {e}")
            # Fallback to regular OpenAI (won't work but prevents crash)
            self.client = OpenAI()

    def ask(self, query: str, max_retries: int = 2) -> str:
        """
        Execute a query against GPT with rules context and word limit enforcement.
        
        Process:
            1. Construct messages with system rules and user query
            2. Send request to GPT API
            3. Check word count of response
            4. If >200 words, request shorter version (up to max_retries)
            5. Truncate to 200 words if still too long after retries
        
        Args:
            query (str): User's natural language question
            max_retries (int): Maximum retry attempts for word count (default: 2)
        
        Returns:
            str: GPT response (max 200 words) or error message
        
        Example:
            >>> client = GPTClient()
            >>> answer = client.ask("What is binning?")
            >>> print(answer)  # Response will be under 200 words
        """
        # ---------------------------------------------------------
        # CONSTRUCT MESSAGES
        # ---------------------------------------------------------
        # System message with rules, followed by user query
        messages = [
            {"role": "system", "content": self.rules},
            {"role": "user", "content": query}
        ]

        # ---------------------------------------------------------
        # QUERY EXECUTION WITH WORD COUNT ENFORCEMENT
        # ---------------------------------------------------------
        for attempt in range(1, max_retries + 1):
            try:
                # Send request to GPT API
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=10  # 10-second timeout
                )
                
                # Extract response text
                text = resp.choices[0].message.content.strip()
                
                # ---------------------------------------------------------
                # WORD COUNT ENFORCEMENT (200-word limit)
                # ---------------------------------------------------------
                word_count = len(text.split())
                if word_count > 200:
                    # Response is too long
                    if attempt < max_retries:
                        # Request shorter version (conversational retry)
                        messages.append({"role": "assistant", "content": text})
                        messages.append({
                            "role": "user", 
                            "content": f"Please provide a shorter response (under 200 words). Current response has {word_count} words."
                        })
                        continue  # Retry with updated conversation
                    else:
                        # Still too long after all retries - force truncate
                        words = text.split()
                        text = " ".join(words[:200]) + "..."
                
                # Return valid response (within word limit)
                return text
                
            except Exception as e:
                # Handle API errors (network, timeout, etc.)
                if attempt >= max_retries:
                    return f"API Error after {max_retries} attempts: {str(e)[:100]}..."
                # Wait before retry
                time.sleep(1)

        # Fallback if no response received
        return "No response received from GPT. Please check connection or credentials."