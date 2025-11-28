import requests
import hashlib
from rdflib import Graph

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
GITHUB_TTL_URL = "https://raw.githubusercontent.com/ifx-dr/DigitalReference/refs/heads/main/DigitalReference.ttl" # URL to download DR TTL
LOCAL_TTL = "DigitalReference.ttl"
LOCAL_HASH_FILE = "DigitalReference.hash" # to store last known hash

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def download_digital_reference(): 
    """Download TTL from GitHub and save locally."""
    print(f"[INFO] Downloading Digital Reference from GitHub...")
    resp = requests.get(GITHUB_TTL_URL) # HTTP GET request

    if resp.status_code != 200: 
        raise Exception(f"[ERROR] Download failed: HTTP {resp.status_code}") # raise error if download fails

    with open(LOCAL_TTL, "wb") as f: # save to local file
        f.write(resp.content) # write content to file
    print("[SUCCESS] Digital Reference downloaded successfully.") # log success message


def compute_hash(path): # Compute SHA256 hash of file
    """Compute SHA256 hash of file."""
    sha = hashlib.sha256() # initialize SHA256 hasher
    with open(path, "rb") as f:
        sha.update(f.read()) # read file and update hash
    return sha.hexdigest() # return hex digest


def check_for_update():
    """Check if TTL changed compared to last run."""
    download_digital_reference() # download latest TTL

    new_hash = compute_hash(LOCAL_TTL) # compute hash of downloaded file

    if not os.path.exists(LOCAL_HASH_FILE): # check if hash file exists
        print("[INFO] No existing hash file - assuming first run.")
        with open(LOCAL_HASH_FILE, "w") as f: # create new hash file
            f.write(new_hash) # write new hash
        return True  # treat as updated

    with open(LOCAL_HASH_FILE, "r") as f: # read existing hash
        old_hash = f.read().strip() 

    if new_hash != old_hash: 
        print("[UPDATE] Digital Reference UPDATED in GitHub!") 
        with open(LOCAL_HASH_FILE, "w") as f: # update hash file
            f.write(new_hash)
        return True # treat as updated
    else:
        print("[INFO] Digital Reference unchanged.")
        return False


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
import os 

if __name__ == "__main__": 
    updated = check_for_update() # check if TTL updated

    print("\n[INFO] Loading Digital Reference into RDF Graph...") 
    g = Graph().parse(LOCAL_TTL) # load TTL into RDF graph
    print(f"[STATS] Total Triples: {len(g)}") # print total triples

    # Quick statistics
    q_classes = """
    PREFIX owl:  <http://www.w3.org/2002/07/owl#> # ontology prefix for classes
    SELECT (COUNT(?c) AS ?n) WHERE { ?c a owl:Class . }
    """
    q_obj = """
    PREFIX owl:  <http://www.w3.org/2002/07/owl#> # ontology prefix for object properties
    SELECT (COUNT(?p) AS ?n) WHERE { ?p a owl:ObjectProperty . }
    """
    q_dat = """ 
    PREFIX owl:  <http://www.w3.org/2002/07/owl#> # ontology prefix for datatype properties
    SELECT (COUNT(?p) AS ?n) WHERE { ?p a owl:DatatypeProperty . }
    """

    for label, q in [("Classes", q_classes), ("ObjectProperties", q_obj), ("DatatypeProperties", q_dat)]: # run queries for stats in loop 
        res = list(g.query(q))[0][0] # get count result
        print(f"{label}: {res}") # print result

    if updated:
        print("\n[ALERT] TTL changed - extraction + embedding steps should be updated.")
    else:
        print("\n[INFO] TTL unchanged - no need to regenerate embeddings.")