"""
Semantic-Condition Transformation (SCT) - Semantic Graph Module
----------------------------------------------------------------

Purpose:
    Generates semantic embeddings for Digital Reference ontology entities.
    Automatically detects when the ontology has been updated on GitHub and
    regenerates embeddings only when needed.

Workflow:
    1. Download latest DigitalReference.ttl from GitHub
    2. Compare hash with previous version to detect changes
    3. If updated, extract ontology entities (Classes, ObjectProperties, DatatypeProperties)
    4. Generate semantic embeddings for each entity's local context
    5. Save embeddings to semantic_conditions.jsonl for use in fusion query system

Features:
    - Auto-update detection via hash comparison
    - Resume support (skips already-processed entities)
    - Rate limiting to respect API quotas
    - Retry logic for failed embedding requests

Output:
    semantic_conditions.jsonl - Contains entity URIs, labels, context text, and embeddings
"""

import os, json, time, hashlib, requests
import httpx, openai, numpy as np
from rdflib import Graph, RDFS, OWL, RDF
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# GitHub repository for Digital Reference ontology
GITHUB_TTL_URL = "https://raw.githubusercontent.com/ifx-dr/DigitalReference/refs/heads/main/DigitalReference.ttl"
LOCAL_TTL = "DigitalReference.ttl"           # Local copy of the ontology file
LOCAL_HASH_FILE = "DigitalReference.hash"    # Hash file to track updates

# GPT4IFX API configuration
BASE_URL  = "https://gpt4ifx.icp.infineon.com"  # Infineon internal LLM endpoint
MODEL     = "sfr-embedding-mistral"              # Embedding model (4096 dimensions)
CERT_PATH = "ca-bundle.crt"                      # SSL certificate for internal network

# Output file for generated semantic embeddings
OUT_FILE = Path("semantic_conditions.jsonl")

# Authentication token (must be manually inserted after running base_file.py)
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJtaWFtIiwiaWF0IjoxNzYwMTkzMzM2LCJpc3MiOiJtaWFtaS1iYXNpYy1hdXRoIiwic3ViIjoiQWlzaHdhcnlhLkFpc2h3YXJ5YUBpbmZpbmVvbi5jb20iLCJhdXRoX3RpbWUiOjE3NjAxOTMzMzYsImNvbXBhbnkiOiJDU0MgRSBJTiIsImdpdmVuX25hbWUiOiJBaXNod2FyeWEiLCJmYW1pbHlfbmFtZSI6IkFpc2h3YXJ5YSIsImVtYWlsIjoiQWlzaHdhcnlhLkFpc2h3YXJ5YUBpbmZpbmVvbi5jb20iLCJ1c2VybmFtZSI6ImFpc2h3YXJ5YSJ9.Zi5EFBHq-lETqNWgMp-911wLkAYQBuO4jyPVf92EG-J23V41uhbi7St0hAXfOS5wV8BSLNcZ47c_kvFpWqn2r2JvN6-Hp91ZibURztSB3RiRLmA55BGG2DexvHlDXvdzJsPLYj3PSnO8mf3LD5EeNVOsYO9S-_4U29ZLN14HGqRjae6e3WRaB6zriRtxLLNO5ag4d1GkLzJdvwdu64495ybapk_iZB08DwOLCoNyQPCcK3AfF6FXWt0-oRiKZo_oRlUUzvihVrqgzMrIaCf3E0btD3OWafPVeRjLdkXkZIPbF0_7d296shu-z2izdeheeXLbuwhRSC3IZ_tbk10Ffg"  

# Rate limiting configuration to avoid API throttling
BATCH_SIZE       = 1                            # Process one entity at a time
MAX_CALLS_MINUTE = 15                           # API quota: 15 requests per minute
SLEEP_PER_CALL   = 60.0 / MAX_CALLS_MINUTE + 0.5  # ~4.5 seconds between calls
RETRY_WAIT       = 10                           # Wait 10 seconds before retry
MAX_RETRIES      = 5                            # Maximum retry attempts for failed requests

# Initialize OpenAI client for GPT4IFX embedding API
client = openai.OpenAI(
    api_key=TOKEN,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=CERT_PATH)
)

# ---------------------------------------------------------
# TTL AUTO-UPDATE DETECTION
# ---------------------------------------------------------
def download_ttl():
    """
    Download the latest Digital Reference TTL file from GitHub.
    Overwrites the local copy with the fresh version.
    """
    print("[INFO] Downloading latest Digital Reference...")
    resp = requests.get(GITHUB_TTL_URL) # HTTP GET request
    if resp.status_code != 200:
        raise Exception("[ERROR] TTL download failed")
    with open(LOCAL_TTL, "wb") as f: # open output file
        f.write(resp.content)
    print("[SUCCESS] TTL downloaded.")

def hash_file(path):
    """
    Compute SHA256 hash of a file to detect content changes.
    
    Args:
        path (str): Path to the file
    
    Returns:
        str: Hexadecimal hash string
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        sha.update(f.read())
    return sha.hexdigest()

def ttl_updated():
    """
    Check if the Digital Reference TTL has been updated on GitHub.
    Compares current hash with previously stored hash.
    
    Returns:
        bool: True if TTL has changed or is first run, False otherwise
    """
    # Download fresh copy from GitHub
    download_ttl()
    new = hash_file(LOCAL_TTL)

    # First run: no previous hash exists
    if not os.path.exists(LOCAL_HASH_FILE):
        print(" First run — treating TTL as updated.")
        open(LOCAL_HASH_FILE, "w").write(new)
        return True

    # Compare with previous hash
    old = open(LOCAL_HASH_FILE).read().strip()
    if new != old:
        print("[UPDATE] TTL UPDATED in GitHub.")
        open(LOCAL_HASH_FILE, "w").write(new)  # Save new hash
        return True
    else:
        print("TTL unchanged since last run.")
        return False


# ---------------------------------------------------------
# MAIN EXECUTION - EMBEDDING GENERATION WITH UPDATE DETECTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # Check if Digital Reference has been updated
    updated = ttl_updated()

    # Exit early if no updates detected (saves computation)
    if not updated:
        print(" No need to regenerate semantic conditions.")
        exit(0)

    # Ask user confirmation before expensive embedding generation
    # (This process can take hours for large ontologies)
    choice = input("\n TTL updated. Regenerate semantic embeddings? (y/n): ").strip().lower()
    if choice != "y":
        print(" Skipping embedding generation.")
        exit(0)

    # Load and parse the RDF graph
    g = Graph().parse(LOCAL_TTL)
    print(f"[STATS] Loaded graph with {len(g)} triples.")

    # Collect all semantic entities from the ontology
    # Includes: OWL Classes, Object Properties, and Datatype Properties
    entities = set(g.subjects(RDF.type, OWL.Class)) \
             | set(g.subjects(RDF.type, OWL.ObjectProperty)) \
             | set(g.subjects(RDF.type, OWL.DatatypeProperty))
    entities = list(entities)
    print(f" Found {len(entities)} entities.")

    # Resume support: load already-processed entity labels
    # Allows script to continue from where it left off if interrupted
    done_labels = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                done_labels.add(rec["label"])
            except:
                pass  # Skip malformed lines

    print(f"[RESUME] Resuming with {len(done_labels)} already processed.")

    # ---------------------------------------------------------
    # HELPER FUNCTIONS FOR GRAPH PROCESSING
    # ---------------------------------------------------------
    
    def get_label(x):
        """
        Extract human-readable label for an entity.
        Falls back to URI fragment if no rdfs:label exists.
        
        Args:
            x: RDF entity (URIRef)
        
        Returns:
            str: Label or URI fragment
        """
        for _, _, lab in g.triples((x, RDFS.label, None)):
            return str(lab)
        # Fallback: extract last part of URI after # or /
        return str(x).split('#')[-1].split('/')[-1]

    def get_local_context(x):
        """
        Retrieve 1-hop neighborhood (local context) of an entity.
        Includes both outgoing and incoming triples.
        
        Args:
            x: RDF entity (URIRef)
        
        Returns:
            list: List of (subject, predicate, object) triples
        """
        triples = []
        # Outgoing edges: (entity, predicate, object)
        for s, p, o in g.triples((x, None, None)):
            triples.append((s, p, o))
        # Incoming edges: (subject, predicate, entity)
        for s, p, o in g.triples((None, None, x)):
            triples.append((s, p, o))
        return triples

    def triples_to_text(tr):
        """
        Convert RDF triples to human-readable text.
        Format: "subject —predicate→ object | ..."
        
        Args:
            tr (list): List of (s, p, o) triples
        
        Returns:
            str: Pipe-separated text representation
        """
        lines = []
        for s, p, o in tr:
            lines.append(f"{get_label(s)} —{get_label(p)}→ {get_label(o)}")
        return " | ".join(lines)

    def safe_embed(t):
        """
        Generate embedding with retry logic for API failures.
        Attempts up to MAX_RETRIES times with exponential backoff.
        
        Args:
            t (str): Text to embed
        
        Returns:
            list or None: Embedding vector or None if all retries failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                return client.embeddings.create(
                    model=MODEL,
                    input=[t],
                    encoding_format="float"
                ).data[0].embedding
            except Exception as e:
                print(f"⚠ Retry {attempt+1}: {e}")
                time.sleep(RETRY_WAIT)
        print("[ERROR] Failed permanently.")
        return None

    # ---------------------------------------------------------
    # MAIN EMBEDDING GENERATION LOOP
    # ---------------------------------------------------------
    # Process each entity: extract context, generate embedding, save to file
    with OUT_FILE.open("a", encoding="utf-8") as fout:
        for entity in tqdm(entities):
            # Get entity label
            label = get_label(entity)
            
            # Skip if already processed (resume support)
            if label in done_labels:
                continue

            # Extract local context (1-hop neighborhood)
            triples = get_local_context(entity)
            if not triples:
                continue  # Skip entities with no connections

            # Convert triples to human-readable text
            context = triples_to_text(triples)
            
            # Generate embedding for the context text
            emb = safe_embed(context)
            if emb is None:
                continue  # Skip if embedding generation failed

            # Write result to JSONL file (one JSON object per line)
            fout.write(json.dumps({
                "entity_uri": str(entity),      # Full URI of the ontology entity
                "label": label,                  # Human-readable label
                "context_text": context,         # Text representation of local graph
                "embedding": emb                 # 4096-dimensional embedding vector
            }, ensure_ascii=False) + "\n")

            # Flush to disk immediately (prevents data loss if interrupted)
            fout.flush()
            done_labels.add(label)
            
            # Rate limiting: sleep to respect API quota
            print(f"Sleeping {SLEEP_PER_CALL:.1f}s…")
            time.sleep(SLEEP_PER_CALL)

    print(" Semantic condition vectors regenerated successfully.")