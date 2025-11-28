"""
GPT Prompt Set - Command-Line Interface
----------------------------------------

Purpose:
    Interactive command-line interface for querying GPT-4o with predefined
    rules and context. Allows users to ask questions with custom system
    prompts loaded from rules.txt.

Usage:
    # With command-line argument:
    python cli.py "What are the temperature specifications?"
    
    # Interactive mode (prompts for question):
    python cli.py

Features:
    - Loads custom rules/context from rules.txt
    - Supports both command-line and interactive input
    - Displays response with word count
    - GPT-4o model with temperature=0.4 (balanced creativity/precision)

Configuration:
    - Model: gpt-4o
    - Temperature: 0.4 (slightly creative but focused responses)
    - Rules file: rules.txt (system prompt/context)

Dependencies:
    - gpt_client.py: GPTClient class for API interaction
    - rules.txt: Custom rules/context for system prompt
"""

import sys
import gpt_client

def main():
    """
    Main CLI function to process user queries with GPT-4o.
    
    Process:
        1. Initialize GPT client with rules.txt and model configuration
        2. Get user question (command-line arg or interactive input)
        3. Send query to GPT-4o with rules context
        4. Display formatted response with word count
        5. Handle errors gracefully
    
    Input Methods:
        - Command-line: python cli.py "your question"
        - Interactive: python cli.py (prompts for input)
    
    Output:
        Formatted GPT response with word count and visual separators
    """
    try:
        # ---------------------------------------------------------
        # INITIALIZE GPT CLIENT
        # ---------------------------------------------------------
        # Load rules.txt for system prompt context
        # Temperature 0.4: balanced between deterministic and creative
        client = gpt_client.GPTClient(rules_path="rules.txt", model="gpt-4o", temperature=0.4)

        # ---------------------------------------------------------
        # GET USER QUESTION
        # ---------------------------------------------------------
        # Check if question provided as command-line argument
        if len(sys.argv) < 2:
            # No argument provided - prompt interactively
            query = input("Your question here please: ").strip()
            
            # Validate input
            if not query:
                print("No question entered. Exiting.")
                sys.exit(1)
        else:
            # Use command-line argument as question
            query = sys.argv[1]

        # ---------------------------------------------------------
        # PROCESS QUERY AND DISPLAY RESPONSE
        # ---------------------------------------------------------
        print("\n================== GPT ANSWER ==================\n")
        
        # Send query to GPT-4o with rules context
        answer = client.ask(query)
        
        # Calculate and display response statistics
        word_count = len(answer.split())
        print(f"[Response: {word_count} words]")
        print()
        
        # Display the response
        print(answer)
        print("\n================================================\n")
        
    except Exception as e:
        # Handle any errors (API failures, file not found, etc.)
        print(f"ERROR: {e}")

# Execute CLI when script is run directly
if __name__ == "__main__":
    main()