"""
RAG Implementation - Intent Classifier for Binning Questions
-------------------------------------------------------------

Purpose:
    Classify user queries to determine if they require binning-specific
    knowledge from research papers. Routes queries to appropriate retrieval
    strategy (RAG vs. general LLM).

Approach:
    Rule-based keyword matching using domain-specific terminology from
    semiconductor manufacturing and testing. Detects all variations of
    "bin" including: bin, bins, binned, binning, etc.

Usage:
    from intent_classifier import is_binning_question
    
    if is_binning_question(user_query):
        # Use RAG retrieval from research papers
    else:
        # Use general LLM or Digital Reference knowledge

Integration:
    Used by sct_fusion_query.py to decide whether to activate conditional
    RAG retrieval for binning-related questions.
"""

import re

def is_binning_question(question: str) -> bool:
    """
    Classify whether a user question is related to binning domain.
    
    Uses rule-based keyword matching to identify binning-related queries
    that require retrieval from research paper corpus. Detects all
    variations of "bin" including binned, bins, binning, etc.
    
    Args:
        question (str): User's natural language query
    
    Returns:
        bool: True if question is binning-related, False otherwise
    
    Examples:
        >>> is_binning_question("What is the bin yield for product X?")
        True
        >>> is_binning_question("The chips are binned into categories")
        True
        >>> is_binning_question("Show me binning results")
        True
        >>> is_binning_question("What are the specifications for temperature?")
        False
        >>> is_binning_question("Combine the results")  # Should not match
        False
    """
    # Convert to lowercase for case-insensitive matching
    q = question.lower()

    # Strong binning-specific keywords (domain terminology)
    # These terms strongly indicate binning-related questions
    strong = [
        "binning",            # Core binning term
        "bin yield",          # Yield metrics per bin
        "test bin",           # Test classification bins
        "binning yield",      # Yield in binning process
        "product bin",        # Product classification bins
        "bin distribution"    # Statistical distribution across bins
    ]

    # Check for strong keyword matches (substring matching)
    if any(w in q for w in strong):
        return True

    # Check for "bin" and its variations (binned, bins, binning) as standalone words
    # Use word boundaries to avoid false positives like "combine", "binary", "cabin"
    # This pattern matches: bin, bins, binned, binning (but not combine, binary, etc.)
    bin_pattern = r'\bbin(?:s|ned|ning)?\b'
    
    if re.search(bin_pattern, q):
        return True

    # No binning-related keywords found
    return False