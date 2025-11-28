"""
Build Unified Corpus from Digital Reference Ontology
-----------------------------------------------------
Purpose:
    Combines extracted ontology components (classes, object properties, 
    datatype properties) into a single unified text corpus for embedding 
    generation.

Inputs:
    - DR_classes.jsonl
    - DR_object_properties.jsonl
    - DR_datatype_properties.jsonl

Output:
    - DR_corpus.jsonl (unified corpus with human-readable text descriptions)

Usage:
    This corpus is then used by build_faiss_index.py to generate embeddings
    for the SCT (Semantic-Condition Transformation) system.
"""

import json

# --- Load all previously generated files ---
with open("DR_classes.jsonl", "r", encoding="utf-8") as f: 
    classes = [json.loads(line) for line in f]

with open("DR_object_properties.jsonl", "r", encoding="utf-8") as f:
    obj_props = [json.loads(line) for line in f]

with open("DR_datatype_properties.jsonl", "r", encoding="utf-8") as f: 
    data_props = [json.loads(line) for line in f]


# --- Build unified text entries for embedding ---
def clean(text): 
    return text.replace("\n", " ").strip() 

corpus = [] 

# 1️⃣  Class definitions
for c in classes:
    txt = f"Class: {c['label']}. Definition: {clean(c['definition']) or 'No definition provided.'}" # Human-readable text
    corpus.append({
        "id": f"class::{c['uri']}", # Unique ID
        "type": "class",
        "text": txt
    })

# 2️⃣  Object properties
for p in obj_props:
    txt = ( 
        f"ObjectProperty: {p['label']} " 
        f"(Domain: {p['domain'] or 'unknown'}, Range: {p['range'] or 'unknown'}). " # Assume unknown if no range
        f"Description: {clean(p['comment']) or 'No comment provided.'}" 
    )
    corpus.append({
        "id": f"objprop::{p['uri']}",
        "type": "object_property",
        "text": txt
    })

# 3️⃣  Datatype properties
for p in data_props:
    txt = (
        f"DatatypeProperty: {p['label']} "
        f"(Domain: {p['domain'] or 'unknown'}, Range: {p['range'] or 'Literal'}). " # Assume Literal if no range
        f"Description: {clean(p['comment']) or 'No comment provided.'}" 
    )
    corpus.append({
        "id": f"dataprop::{p['uri']}", 
        "type": "datatype_property", 
        "text": txt 
    })


# --- Save unified corpus ---
with open("DR_corpus.jsonl", "w", encoding="utf-8") as f: # Save as JSONL
    for entry in corpus:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Unified corpus built successfully: {len(corpus)} entries")
print("Saved as DR_corpus.jsonl")