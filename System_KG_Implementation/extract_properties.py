import os
import json
import requests
import hashlib # for SHA256 hashing
from rdflib import Graph # for RDF parsing

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
GITHUB_TTL_URL = "https://raw.githubusercontent.com/ifx-dr/DigitalReference/refs/heads/main/DigitalReference.ttl"
LOCAL_TTL = "DigitalReference.ttl"
LOCAL_HASH_FILE = "DigitalReference.hash"

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def download_ttl():
    print("[INFO] Downloading latest Digital Reference...") 
    resp = requests.get(GITHUB_TTL_URL) # HTTP GET request
    if resp.status_code != 200: # check if download failed
        raise Exception(f"[ERROR] Download failed ({resp.status_code})") # raise error if download fails
    with open(LOCAL_TTL, "wb") as f: # open output file
        f.write(resp.content) # write content to file
    print("[SUCCESS] TTL downloaded.") 

def hash_file(path): # Compute SHA256 hash of file
    sha = hashlib.sha256() # initialize SHA256 hasher
    with open(path, "rb") as f: # open file in binary mode
        sha.update(f.read()) # update hash with file content
    return sha.hexdigest() # return hex digest

def ttl_updated(): # Check if TTL file has been updated
    download_ttl()
    new = hash_file(LOCAL_TTL) # compute hash of downloaded file

    if not os.path.exists(LOCAL_HASH_FILE): # check if hash file exists
        with open(LOCAL_HASH_FILE, "w") as f: # open hash file for writing
            f.write(new) # write new hash
        return True

    old = open(LOCAL_HASH_FILE).read().strip() # read existing hash
    if new != old:
        print("[UPDATE] TTL UPDATED - will re-extract DR properties.")
        open(LOCAL_HASH_FILE, "w").write(new) # update hash file with new hash
        return True
    else:
        print(" TTL unchanged — skipping extraction.") 
        return False                                                                                   


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__": 
    needs_extract = ttl_updated() # check if TTL updated

    if not needs_extract:
        exit(0)

    print("[LOAD] Parsing DigitalReference.ttl ...") 
    g = Graph().parse(LOCAL_TTL) 

    # ----- OBJECT PROPERTIES -----
    q_obj = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> # RDF Schema prefix for labels and comments
    PREFIX owl:  <http://www.w3.org/2002/07/owl#> # OWL prefix for classes
    SELECT ?prop ?label ?domain ?range ?comment WHERE { # query for object properties
      ?prop a owl:ObjectProperty .
      OPTIONAL { ?prop rdfs:label ?label }
      OPTIONAL { ?prop rdfs:domain ?domain }
      OPTIONAL { ?prop rdfs:range  ?range }
      OPTIONAL { ?prop rdfs:comment ?comment }
    }
    """

    object_properties = [] # to hold object property records
    for row in g.query(q_obj): # iterate over query results
        prop_uri = str(row.prop) # property URI
        label = str(row.label) if row.label else prop_uri.split("#")[-1].split("/")[-1] 
        domain = str(row.domain) if row.domain else "" 
        range_ = str(row.range) if row.range else "" 
        comment = str(row.comment) if row.comment else "" 
        object_properties.append({
            "uri": prop_uri, 
            "label": label,
            "domain": domain,
            "range": range_,
            "comment": comment
        })

    with open("DR_object_properties.jsonl", "w", encoding="utf-8") as f: # open output file
        for rec in object_properties: # iterate over object property records
            f.write(json.dumps(rec, ensure_ascii=False) + "\n") # write each record as JSON line

    print(f"[SUCCESS] Saved {len(object_properties)} object properties.")

    # ----- DATATYPE PROPERTIES -----
    q_data = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl:  <http://www.w3.org/2002/07/owl#> 
    SELECT ?prop ?label ?domain ?range ?comment WHERE {
      ?prop a owl:DatatypeProperty .
      OPTIONAL { ?prop rdfs:label ?label }
      OPTIONAL { ?prop rdfs:domain ?domain }
      OPTIONAL { ?prop rdfs:range  ?range }
      OPTIONAL { ?prop rdfs:comment ?comment }
    }
    """

    datatype_properties = []
    for row in g.query(q_data): # iterate over query results
        prop_uri = str(row.prop) # property URI
        label = str(row.label) if row.label else prop_uri.split("#")[-1].split("/")[-1] 
        domain = str(row.domain) if row.domain else "" 
        range_ = str(row.range) if row.range else "" 
        comment = str(row.comment) if row.comment else "" 
        datatype_properties.append({
            "uri": prop_uri,
            "label": label,
            "domain": domain,
            "range": range_,
            "comment": comment
        })

    with open("DR_datatype_properties.jsonl", "w", encoding="utf-8") as f: # open output file
        for rec in datatype_properties: # iterate over datatype property records
            f.write(json.dumps(rec, ensure_ascii=False) + "\n") # write each record as JSON line

    print(f"[SUCCESS] Saved {len(datatype_properties)} datatype properties.")
    print(" Extraction complete.")