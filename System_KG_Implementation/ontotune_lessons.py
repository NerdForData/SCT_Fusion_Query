"""
OntoTune Step 4.2 – Micro-Lesson Generator
------------------------------------------

Goal:
  For each NEW DR concept (detected by ontotune_detect.py),
  build a compact 'micro lesson' using its local graph context.

Inputs:
  - DigitalReference.ttl
  - new_concepts.jsonl   (from ontotune_detect.py)

Output:
  - ontotune_lessons.jsonl   (one JSONL entry per concept)
"""

import json
import time # for throttling and delays
from pathlib import Path
import httpx
import openai 
from rdflib import Graph, RDFS, RDF, OWL # for RDF parsing

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
TTL_FILE = "DigitalReference.ttl"
NEW_CONCEPTS_FILE = Path("new_concepts.jsonl")
LESSONS_FILE = Path("ontotune_lessons.jsonl")

BASE_URL  = "https://gpt4ifx.icp.infineon.com"
MODEL_LLM = "gpt-4o"
CERT_PATH = "ca-bundle.crt"

#  Fill this with a valid GPT4IFX bearer token
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJtaWFtIiwiaWF0IjoxNzYwMTkzMzM2LCJpc3MiOiJtaWFtaS1iYXNpYy1hdXRoIiwic3ViIjoiQWlzaHdhcnlhLkFpc2h3YXJ5YUBpbmZpbmVvbi5jb20iLCJhdXRoX3RpbWUiOjE3NjAxOTMzMzYsImNvbXBhbnkiOiJDU0MgRSBJTiIsImdpdmVuX25hbWUiOiJBaXNod2FyeWEiLCJmYW1pbHlfbmFtZSI6IkFpc2h3YXJ5YSIsImVtYWlsIjoiQWlzaHdhcnlhLkFpc2h3YXJ5YUBpbmZpbmVvbi5jb20iLCJ1c2VybmFtZSI6ImFpc2h3YXJ5YSJ9.Zi5EFBHq-lETqNWgMp-911wLkAYQBuO4jyPVf92EG-J23V41uhbi7St0hAXfOS5wV8BSLNcZ47c_kvFpWqn2r2JvN6-Hp91ZibURztSB3RiRLmA55BGG2DexvHlDXvdzJsPLYj3PSnO8mf3LD5EeNVOsYO9S-_4U29ZLN14HGqRjae6e3WRaB6zriRtxLLNO5ag4d1GkLzJdvwdu64495ybapk_iZB08DwOLCoNyQPCcK3AfF6FXWt0-oRiKZo_oRlUUzvihVrqgzMrIaCf3E0btD3OWafPVeRjLdkXkZIPbF0_7d296shu-z2izdeheeXLbuwhRSC3IZ_tbk10Ffg"

SLEEP_BETWEEN_CALLS = 2.0   # seconds – new concepts are few, so light throttling


# ---------------------------------------------------------
# GPT4IFX CLIENT
# ---------------------------------------------------------
client = openai.OpenAI(
    api_key=TOKEN,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=CERT_PATH)
)


# ---------------------------------------------------------
# LOAD GRAPH
# ---------------------------------------------------------
print(" Loading DigitalReference.ttl …")
g = Graph().parse(TTL_FILE)
print(f" Graph loaded with {len(g)} triples")


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def get_label(entity):
    """Return rdfs:label if present, otherwise local name."""
    for _, _, lab in g.triples((entity, RDFS.label, None)): # check for rdfs:label
        return str(lab) # return label if found
    return str(entity).split("#")[-1].split("/")[-1] # fallback to local name


def get_local_context(entity_uri, depth=1): # depth=1 for 1-hop neighborhood
    """Collect outgoing & incoming triples (1-hop neighborhood)."""
    entity = g.resource(entity_uri) # get rdflib resource
    e = entity.identifier # get URIRef

    visited = {e} # track visited nodes
    frontier = {e} # nodes to explore
    triples = [] # collected triples

    for _ in range(depth): # for each depth level
        next_frontier = set() # nodes to explore next
        # outgoing
        for s, p, o in g.triples((None, None, None)): # iterate over all triples
            if s == e and (s, p, o) not in triples: # outgoing triple from entity
                triples.append((s, p, o)) # add outgoing triple
                next_frontier.add(o) # add object to next frontier
            if o == e and (s, p, o) not in triples: # incoming triple to entity
                triples.append((s, p, o)) # add incoming triple
                next_frontier.add(s) # add subject to next frontier
        frontier = next_frontier - visited # update frontier
        visited |= frontier # mark new nodes as visited

    return triples # return collected triples


def triples_to_text(triples):
    """Convert triples to a compact DR-context string.""" 
    if not triples:
        return "" 
    parts = [] # to hold triple strings
    for s, p, o in triples:
        s_lab = get_label(s) # get subject label
        p_lab = get_label(p) # get predicate label
        o_lab = get_label(o) # get object label
        parts.append(f"{s_lab} —{p_lab}→ {o_lab}") # format triple 
    return " | ".join(parts) # join all triples


def call_llm_for_lesson(label, uri, concept_type, dr_context_text): 
    """
    Ask GPT-4IFX to turn the DR context into a micro-lesson.
    Returns a dict with fields: definition, engineering_note, dr_context.
    """
    # Fallback DR context if empty
    if not dr_context_text: # if no context found
        dr_context_text = f"{label} —type→ {concept_type}" # minimal context

    system_prompt = (
        "You are a semiconductor and ontology tutor. "
        "Given a concept from a Digital Reference (knowledge graph) and its local relations, "
        "write a very compact 'micro lesson' that teaches this concept."
    )

    user_prompt = f"""
Concept label: {label} 
Concept URI: {uri}
Concept type: {concept_type}
Digital Reference local context:
{dr_context_text}

Please respond ONLY in valid JSON with the following keys:
- "definition": one or two sentences explaining what this concept means.
- "engineering_note": one or two sentences explaining why this concept matters in semiconductor engineering or manufacturing.
- "dr_context_short": a shortened version of the context (max ~200 characters) that preserves the most relevant relations.

Example JSON:
{{
  "definition": "...",
  "engineering_note": "...",
  "dr_context_short": "A —affects→ B | B —partOf→ C"
}}
"""

    try:
        resp = client.chat.completions.create( # call GPT4IFX
            model=MODEL_LLM,
            temperature=0.2, # low temperature for factuality
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content.strip() # get response text
        # Try to parse JSON from response
        data = json.loads(text) # parse JSON response
        return {
            "definition": data.get("definition", "").strip(), # get definition
            "engineering_note": data.get("engineering_note", "").strip(), # get engineering note
            "dr_context_short": data.get("dr_context_short", dr_context_text[:200]), # get shortened context
        }
    except Exception as e:
        print(f" LLM call failed for {label}: {e}") # log error
        # Fallback: simple rule-based micro lesson
        return {
            "definition": f"{label} is a concept from the Digital Reference ontology (type: {concept_type}).", # simple definition
            "engineering_note": ( # simple engineering note
                f"This concept participates in semiconductor-related relations "
                f"described in the Digital Reference and may influence process, product or yield."
            ),
            "dr_context_short": dr_context_text[:200], 
        }


# ---------------------------------------------------------
# LOAD NEW CONCEPTS
# ---------------------------------------------------------
if not NEW_CONCEPTS_FILE.exists(): # check file existence
    print(" new_concepts.jsonl not found – nothing to do.")
    exit(0)

new_concepts = [] # to hold new concept records
with NEW_CONCEPTS_FILE.open(encoding="utf-8") as f: # open file
    for line in f: # read line by line
        line = line.strip() 
        if not line: # skip empty lines
            continue
        try:
            new_concepts.append(json.loads(line)) 
        except Exception as e: # log parse errors
            print(f" Skipping invalid line in new_concepts.jsonl: {e}")

if not new_concepts:
    print(" No new concepts found in new_concepts.jsonl.") 
    exit(0) # exit if no concepts

print(f" Found {len(new_concepts)} new concepts to generate lessons for.")

# Support re-running: skip concepts already in lessons file
existing_uris = set() # to track existing lessons
if LESSONS_FILE.exists(): 
    with LESSONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                existing_uris.add(rec.get("uri")) # track existing URIs
            except:
                pass
    print(f" Resuming – {len(existing_uris)} lessons already exist.") # log count for resuming 

# ---------------------------------------------------------
# MAIN LOOP – BUILD MICRO-LESSONS
# ---------------------------------------------------------
with LESSONS_FILE.open("a", encoding="utf-8") as fout: # append mode
    for rec in new_concepts: # iterate over new concepts
        uri = rec["uri"] # get concept URI
        ctype = rec["type"] # get concept type

        if uri in existing_uris:
            print(f" Skipping {uri} (lesson already exists)") # skip if lesson exists
            continue

        print(f"\n Building micro-lesson for {ctype}: {uri}") 

        # Local DR context
        triples = get_local_context(uri, depth=1) # get local context triples
        dr_context_text = triples_to_text(triples)

        # Label
        label = get_label(g.resource(uri).identifier) # get concept label

        # Call GPT-4IFX (or fallback)
        lesson = call_llm_for_lesson(label, uri, ctype, dr_context_text) # call LLM for lesson

        # Compose final record
        out = {
            "uri": uri,
            "type": ctype,
            "label": label,
            "definition": lesson["definition"],
            "engineering_note": lesson["engineering_note"],
            "dr_context": lesson["dr_context_short"],
        }

        fout.write(json.dumps(out, ensure_ascii=False) + "\n") # write to file the new lesson
        fout.flush() # ensure it's written
        print(f" Saved micro-lesson for: {label}") # log saved lesson
        time.sleep(SLEEP_BETWEEN_CALLS)

print("\n OntoTune Step 4.2 complete!")
print(f" Lessons saved in: {LESSONS_FILE.resolve()}")