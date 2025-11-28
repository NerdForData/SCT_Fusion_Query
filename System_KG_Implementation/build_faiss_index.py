"""
Build FAISS index for Digital Reference ontology using Infineon GPT-4IFX embeddings API.

Steps:
1. Choose authentication method:
   - Basic Auth (service account username/password)
   - Bearer Auth (token copied from /auth/token web page)
2. Generate embeddings for DR_corpus.jsonl entries using sfr-embedding-mistral
3. Save FAISS index (DR_faiss.index) and ID mapping (DR_ids.npy)
"""

import base64
import httpx
import openai
import json
import numpy as np
import faiss 
import time
from tqdm import tqdm # progress bar

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
BASE_URL = "https://gpt4ifx.icp.infineon.com"
MODEL = "sfr-embedding-mistral" # embedding model
CERT_PATH = "ca-bundle.crt" # certificate

# --- Choose authentication mode ---
basic_auth = True # set False if using bearer token
USERNAME = "INFINEON\\Aishwarya"      
PASSWORD = "Gauri@123456789"        
BEARER_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJtaWFtIiwiaWF0IjoxNzYwMTkzMzM2LCJpc3MiOiJtaWFtaS1iYXNpYy1hdXRoIiwic3ViIjoiQWlzaHdhcnlhLkFpc2h3YXJ5YUBpbmZpbmVvbi5jb20iLCJhdXRoX3RpbWUiOjE3NjAxOTMzMzYsImNvbXBhbnkiOiJDU0MgRSBJTiIsImdpdmVuX25hbWUiOiJBaXNod2FyeWEiLCJmYW1pbHlfbmFtZSI6IkFpc2h3YXJ5YSIsImVtYWlsIjoiQWlzaHdhcnlhLkFpc2h3YXJ5YUBpbmZpbmVvbi5jb20iLCJ1c2VybmFtZSI6ImFpc2h3YXJ5YSJ9.Zi5EFBHq-lETqNWgMp-911wLkAYQBuO4jyPVf92EG-J23V41uhbi7St0hAXfOS5wV8BSLNcZ47c_kvFpWqn2r2JvN6-Hp91ZibURztSB3RiRLmA55BGG2DexvHlDXvdzJsPLYj3PSnO8mf3LD5EeNVOsYO9S-_4U29ZLN14HGqRjae6e3WRaB6zriRtxLLNO5ag4d1GkLzJdvwdu64495ybapk_iZB08DwOLCoNyQPCcK3AfF6FXWt0-oRiKZo_oRlUUzvihVrqgzMrIaCf3E0btD3OWafPVeRjLdkXkZIPbF0_7d296shu-z2izdeheeXLbuwhRSC3IZ_tbk10Ffg"

# ------------------------------------------------------------
# AUTHENTICATION SETUP
# ------------------------------------------------------------
def generate_base64_string(username, password): 
    """Encode username:password to base64"""
    return base64.b64encode(f"{username}:{password}".encode("ascii")).decode("ascii") 

if basic_auth:
    token = generate_base64_string(USERNAME, PASSWORD) 
    headers = {"Authorization": f"Basic {token}"} 
else:
    token = BEARER_TOKEN
    headers = {"Authorization": f"Bearer {token}"}

# ------------------------------------------------------------
# CREATE CLIENT
# ------------------------------------------------------------
client = openai.OpenAI(
    api_key=token,
    base_url=BASE_URL,
    default_headers=headers,
    http_client=httpx.Client(verify=CERT_PATH)
)

# ------------------------------------------------------------
# LOAD CORPUS
# ------------------------------------------------------------  
with open("DR_corpus.jsonl", "r", encoding="utf-8") as f:
    corpus = [json.loads(line) for line in f] # load all lines as JSON objects

texts = [c["text"] for c in corpus] # keeps track of texts
ids = [c["id"] for c in corpus] # keep track of IDs
print(f"[LOAD] Loaded {len(texts)} entries from DR_corpus.jsonl")

# ------------------------------------------------------------
# GENERATE EMBEDDINGS
# ------------------------------------------------------------
vectors = [] # list to hold embedding vectors
BATCH_SIZE = 128 # batch size for embedding requests
print(f"[START] Generating embeddings with model: {MODEL}") # log start message

for i in tqdm(range(0, len(texts), BATCH_SIZE)): # process in batches
    batch = texts[i:i + BATCH_SIZE] # current batch of texts
    for attempt in range(3): # retry up to 3 times
        try:
            embed = client.embeddings.create( # embedding request
                model=MODEL,
                input=batch,
                encoding_format="float"
            )
            vectors.extend([np.array(e.embedding, dtype="float32") for e in embed.data]) # add to vectors list 
            break
        except Exception as e:
            print(f"[WARNING] Attempt {attempt+1}/3 failed at batch {i}: {e}") # log warning message
            time.sleep(5)
    else:
        print(f"[ERROR] Skipping batch {i} after 3 failed attempts")

# -----------------------------------------------------------
# BUILD & SAVE FAISS INDEX
# ------------------------------------------------------------
if not vectors:
    raise SystemExit("[ERROR] No embeddings generated. Check credentials or network/VPN.") # exit if no embeddings

emb_matrix = np.vstack(vectors) # convert list of vectors to matrix
print("[SUCCESS] Embedding matrix shape:", emb_matrix.shape) # log matrix shape

faiss.normalize_L2(emb_matrix) # normalize embeddings
index = faiss.IndexFlatIP(emb_matrix.shape[1]) # create index
index.add(emb_matrix) # add embeddings to index
faiss.write_index(index, "DR_faiss.index") # save index to file
np.save("DR_ids.npy", np.array(ids)) # save IDs to file
print("\n[SUCCESS] SUCCESS! Internal FAISS index built and saved.")
print("   Files created:")
print("   - DR_faiss.index")
print("   - DR_ids.npy")