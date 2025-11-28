from rdflib import Graph # for RDF parsing
import json

# Load your existing ontology
g = Graph() # initialize RDF graph
g.parse("DigitalReference.ttl") # parse TTL into graph

# SPARQL to fetch all classes with label + definition/comment if any
q = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> # RDF Schema prefix for labels and comments
PREFIX owl:  <http://www.w3.org/2002/07/owl#> # OWL prefix for classes
PREFIX skos: <http://www.w3.org/2004/02/skos/core#> # SKOS prefix for definitions

SELECT ?cls ?label ?def WHERE {
  ?cls a owl:Class .
  OPTIONAL { ?cls rdfs:label ?label } # optional label
  OPTIONAL { ?cls rdfs:comment ?def } # optional comment as definition
  OPTIONAL { ?cls skos:definition ?def } # optional SKOS definition
}
"""

records = [] # to hold class records
for row in g.query(q): # iterate over query results
    uri = str(row.cls) # class URI
    label = str(row.label) if row.label else uri.split('#')[-1].split('/')[-1] # fallback label
    definition = str(row['def']) if row['def'] else "" # fallback definition
    records.append({ # create record
        "uri": uri,
        "label": label,
        "definition": definition
    })

print(f"Extracted {len(records)} classes") # print number of extracted classes

# Save as JSONL (each line = one class)
with open("DR_classes.jsonl", "w", encoding="utf-8") as f: # open output file
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n") # write each record as JSON line

print("Saved to DR_classes.jsonl") 