"""
RAG Implementation - Embedding Generation for Research Paper Chunks
--------------------------------------------------------------------

Purpose:
    Generate semantic embeddings for binning research paper chunks extracted
    from PDF files. Creates a FAISS vector index for fast similarity search.

Workflow:
    1. Load text chunks from rag_chunks.jsonl (output of load_pdfs.py)
    2. Generate embeddings using GPT4IFX sfr-embedding-mistral model
    3. Build FAISS index with normalized vectors for cosine similarity
    4. Save index and document metadata for retrieval

Features:
    - Checkpoint/resume support (saves progress every 10 chunks)
    - Automatic token refresh every 500 embeddings (prevents expiration)
    - Error handling with retry logic
    - Rate limiting (15 requests/minute)
    - Zero-vector fallback for failed embeddings

Inputs:
    rag_chunks.jsonl - Text chunks from research papers

Outputs:
    binning_faiss.index - FAISS vector index (4096-dimensional)
    binning_docs.jsonl  - Document metadata for retrieval
"""

import json, time
from pathlib import Path
import numpy as np
import faiss
import openai, httpx
import base64, requests, urllib3

# Disable SSL warnings for internal Infineon network
urllib3.disable_warnings()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# GPT4IFX API settings
BASE_URL = "https://gpt4ifx.icp.infineon.com"  # Infineon internal LLM endpoint
MODEL_EMB = "sfr-embedding-mistral"             # Embedding model (4096 dimensions)
CERT_PATH = Path(__file__).parent.parent / "ca-bundle.crt"  # SSL certificate

# Authentication credentials
USERNAME = "INFINEON\\Aishwarya"
PASSWORD = "Gauri@123456789"

# File paths (relative to parent directory)
IN_FILE = Path(__file__).parent.parent / "System_KG_Implementation" / "rag_chunks.jsonl"  # Input: text chunks from PDFs
OUT_INDEX = "binning_faiss.index"                             # Output: FAISS vector index
OUT_DOCS  = "binning_docs.jsonl"                              # Output: document metadata
CHECKPOINT_FILE = Path(__file__).parent / "embedding_checkpoint.json"  # Resume state

# Rate limiting to respect API quota
MAX_CALLS_MIN = 15              # Maximum 15 requests per minute
SLEEP = 60/MAX_CALLS_MIN + 0.5  # ~4.5 seconds between requests

# ---------------------------------------------------------
# AUTHENTICATION FUNCTIONS
# ---------------------------------------------------------

def get_token():
    """
    Authenticate with GPT4IFX and retrieve Bearer token.
    Uses Basic Auth to exchange credentials for session token.
    
    Returns:
        str: Bearer token for API authentication
    """
    # Encode credentials as Base64 for Basic Auth
    token = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
    headers = {"Authorization": f"Basic {token}"}
    
    # Request Bearer token from authentication endpoint
    r = requests.get(f"{BASE_URL}/auth/token", headers=headers, verify=False)
    return r.headers.get("x-forwarded-access-token")

def refresh_client():
    """
    Refresh the OpenAI client with a new authentication token.
    
    Important: Tokens expire after ~1.5 hours, so this function should be
    called periodically during long-running embedding generation processes.
    
    Updates the global 'client' variable with a fresh token.
    """
    global client
    client = openai.OpenAI(
        api_key=get_token(),
        base_url=BASE_URL,
        http_client=httpx.Client(verify=False),
    )

# Initialize client with fresh token
refresh_client()

# ---------------------------------------------------------
# EMBEDDING AND CHECKPOINT FUNCTIONS
# ---------------------------------------------------------

def embed_text(t):
    """
    Generate embedding vector for a given text string.
    
    Args:
        t (str): Text to embed
    
    Returns:
        np.ndarray: 4096-dimensional embedding vector (float32)
    """
    r = client.embeddings.create(
        model=MODEL_EMB,
        input=[t],
        encoding_format="float"
    )
    return np.array(r.data[0].embedding, dtype="float32")

def save_checkpoint(index, vectors):
    """
    Save progress checkpoint to enable resume after interruption.
    Saves both the current index position and all generated vectors.
    
    Args:
        index (int): Last successfully processed chunk index
        vectors (list): List of embedding vectors generated so far
    """
    # Save metadata (position in file)
    checkpoint = {
        "last_index": index,
        "vectors_count": len(vectors)
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)
    
    # Save all vectors as numpy array for efficient loading
    if vectors:
        np.save(CHECKPOINT_FILE.with_suffix(".npy"), np.vstack(vectors))

def load_checkpoint():
    """
    Load progress checkpoint if exists, allowing resume from interruption.
    
    Returns:
        tuple: (last_processed_index, list_of_vectors)
               Returns (-1, []) if no checkpoint exists
    """
    if CHECKPOINT_FILE.exists():
        # Load checkpoint metadata
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
        
        # Load saved vectors
        vectors_file = CHECKPOINT_FILE.with_suffix(".npy")
        if vectors_file.exists():
            # Load the stacked matrix and convert back to list of vectors
            matrix = np.load(vectors_file)
            vectors = [matrix[i] for i in range(len(matrix))]
            print(f"[RESUME] Resuming from chunk {checkpoint['last_index'] + 1}")
            print(f"[RESUME] Loaded {len(vectors)} existing embeddings")
            return checkpoint['last_index'], vectors
    
    # No checkpoint found - start fresh
    return -1, []

# ---------------------------------------------------------
# MAIN EMBEDDING GENERATION PROCESS
# ---------------------------------------------------------

def main():
    """
    Main function to process all text chunks and generate embeddings.
    
    Process:
        1. Load checkpoint (if exists) to resume from interruption
        2. Load all text chunks from input file
        3. Generate embeddings with automatic token refresh
        4. Handle errors with retry logic and zero-vector fallback
        5. Save checkpoints every 10 chunks
        6. Build FAISS index and save outputs
    """
    docs, vectors = [], []
    
    # Load checkpoint if exists (resume support)
    start_index, vectors = load_checkpoint()

    # Load all text chunks from input file
    with IN_FILE.open(encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj)

    print(f"[LOAD] Loaded {len(docs)} chunks")
    
    # Validate input
    if len(docs) == 0:
        print("[ERROR] No chunks found. Run load_pdfs.py first to generate chunks.")
        return
    
    # Display resume status
    if start_index >= 0:
        print(f"[RESUME] Resuming from chunk {start_index + 1}, already have {len(vectors)} embeddings")
    else:
        print("[START] Starting fresh embedding process")

    # ---------------------------------------------------------
    # EMBEDDING GENERATION LOOP
    # ---------------------------------------------------------
    for i, d in enumerate(docs):
        # Skip already processed chunks (resume support)
        if i <= start_index:
            continue
            
        print(f"Embedding {i+1}/{len(docs)}")
        
        # Refresh authentication token every 500 embeddings
        # Tokens expire after ~1.5 hours, so this prevents expiration during long runs
        if i > 0 and i % 500 == 0:
            print("[AUTH] Refreshing authentication token...")
            refresh_client()

        try:
            # Generate embedding for current chunk
            v = embed_text(d["text"])
            vectors.append(v)
            
        except openai.BadRequestError as e:
            # Handle bad request errors (often due to expired token)
            print(f"[WARNING] Bad request for chunk {i+1} (source: {d.get('source', 'unknown')}): {str(e)[:100]}")
            
            # Try refreshing token and retry once
            print("[AUTH] Refreshing token and retrying...")
            refresh_client()
            time.sleep(2)
            
            try:
                v = embed_text(d["text"])
                vectors.append(v)
                print(f"[SUCCESS] Retry successful for chunk {i+1}")
            except Exception as retry_error:
                print(f"[ERROR] Retry failed for chunk {i+1}: {str(retry_error)[:100]}")
                # Add zero vector as placeholder to maintain alignment
                vectors.append(np.zeros(4096, dtype="float32"))
                
        except Exception as e:
            # Handle other errors (network issues, API errors, etc.)
            print(f"[WARNING] Error for chunk {i+1}: {str(e)[:100]}")
            # Add zero vector as placeholder
            vectors.append(np.zeros(4096, dtype="float32"))

        # Save checkpoint every 10 chunks to prevent data loss
        if (i + 1) % 10 == 0:
            save_checkpoint(i, vectors)
            print(f"[CHECKPOINT] Checkpoint saved at {i+1}/{len(docs)}")

        # Rate limiting: sleep between requests to respect API quota
        time.sleep(SLEEP)      # ~4.5 seconds per request

    print(f"\n[SUCCESS] Completed all embeddings!")
    
    # ---------------------------------------------------------
    # CLEANUP AND INDEX CREATION
    # ---------------------------------------------------------
    
    # Clean up checkpoint files (no longer needed)
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    if CHECKPOINT_FILE.with_suffix(".npy").exists():
        CHECKPOINT_FILE.with_suffix(".npy").unlink()

    # Stack all vectors into a matrix and normalize for cosine similarity
    emb_matrix = np.vstack(vectors)
    faiss.normalize_L2(emb_matrix)  # L2 normalization for Inner Product = cosine similarity

    # Create FAISS index for fast similarity search
    # IndexFlatIP = Inner Product index (works with normalized vectors for cosine similarity)
    index = faiss.IndexFlatIP(emb_matrix.shape[1])  # 4096 dimensions
    index.add(emb_matrix)
    
    # Save FAISS index to disk
    faiss.write_index(index, OUT_INDEX)
    print("[SAVE] Saved binning_faiss.index")

    # Save document metadata (aligned with vectors by position)
    with open(OUT_DOCS, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("[SAVE] Saved binning_docs.jsonl")

if __name__ == "__main__":
    main()