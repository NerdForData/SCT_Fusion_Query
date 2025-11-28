"""
OntoTune Step 4.1 – DR Change Detector (fixed version)

Goal:
  - Compare the CURRENT DigitalReference.ttl with the last baseline
    stored in previous_entities.json.
  - Find ONLY truly new DR concepts (classes, object properties,
    datatype properties).
  - Exclude any concepts that already have micro-lessons
    in ontotune_lessons.jsonl.
  - Write those concepts to new_concepts.jsonl for OntoTune Step 4.2.
"""

import json
from pathlib import Path

from rdflib import Graph, RDF, OWL

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TTL_FILE = "DigitalReference.ttl"

PREVIOUS_FILE = Path("previous_entities.json")
NEW_CONCEPTS_FILE = Path("new_concepts.jsonl")
LESSONS_FILE = Path("ontotune_lessons.jsonl")

# Restrict to DR namespace only (avoid external vocabularies)
DR_PREFIXES = (
    "http://www.w3id.org/ecsel-dr",
    "https://www.w3id.org/ecsel-dr",
    "http://w3id.org/ecsel-dr",
    "https://w3id.org/ecsel-dr",
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def is_dr_uri(uri: str) -> bool:
    """Return True if the URI belongs to Digital Reference namespace."""
    if uri.startswith("_:"):
        # blank node -> ignore for new concept detection
        return False
    return uri.startswith(DR_PREFIXES)


def collect_current_entities(g: Graph):
    """Collect DR-only classes, object properties, datatype properties."""
    classes = set()
    obj_props = set()
    data_props = set()

    for c in g.subjects(RDF.type, OWL.Class):
        u = str(c)
        if is_dr_uri(u):
            classes.add(u)

    for p in g.subjects(RDF.type, OWL.ObjectProperty):
        u = str(p)
        if is_dr_uri(u):
            obj_props.add(u)

    for p in g.subjects(RDF.type, OWL.DatatypeProperty):
        u = str(p)
        if is_dr_uri(u):
            data_props.add(u)

    return classes, obj_props, data_props


def load_previous_entities():
    """Load baseline entities from previous_entities.json (if exists)."""
    if not PREVIOUS_FILE.exists():
        return set(), set(), set()

    with PREVIOUS_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    return (
        set(data.get("classes", [])),
        set(data.get("object_properties", [])),
        set(data.get("datatype_properties", [])),
    )


def load_existing_lesson_uris():
    """All URIs that already have a micro-lesson."""
    if not LESSONS_FILE.exists():
        return set()

    uris = set()
    with LESSONS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                u = rec.get("uri")
                if u:
                    uris.add(u)
            except Exception:
                # ignore malformed lines
                continue
    return uris


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading DigitalReference.ttl ...")
    g = Graph().parse(TTL_FILE)
    print(f"Loaded graph with {len(g)} triples")

    # Current snapshot
    curr_classes, curr_obj, curr_data = collect_current_entities(g)
    print("Current DR entities (filtered):")
    print(f"   DR Classes           : {len(curr_classes)}")
    print(f"   DR Object Properties : {len(curr_obj)}")
    print(f"   DR Datatype Properties: {len(curr_data)}")

    # Baseline snapshot
    prev_classes, prev_obj, prev_data = load_previous_entities()

    first_run = len(prev_classes) == 0 and len(prev_obj) == 0 and len(prev_data) == 0

    # URIs that already have OntoTune lessons
    lesson_uris = load_existing_lesson_uris()
    if lesson_uris:
        print(f"Existing micro-lessons found for {len(lesson_uris)} URIs.")

    # -----------------------------------------------------
    # DETECT NEW CONCEPTS
    # -----------------------------------------------------
    if first_run:
        print("First run detected – creating baseline ONLY (no new concepts).")

        baseline = {
            "classes": sorted(curr_classes),
            "object_properties": sorted(curr_obj),
            "datatype_properties": sorted(curr_data),
        }
        with PREVIOUS_FILE.open("w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)

        # Overwrite / create empty new_concepts.jsonl
        NEW_CONCEPTS_FILE.write_text("", encoding="utf-8")
        print("Baseline written to previous_entities.json")
        print("Empty new_concepts.jsonl created.")
    else:
        # Compute diff vs baseline
        new_classes = curr_classes - prev_classes
        new_obj = curr_obj - prev_obj
        new_data = curr_data - prev_data

        print("Detected NEW DR entities before lesson-filtering:")
        print(f"   New Classes           : {len(new_classes)}")
        print(f"   New Object Properties : {len(new_obj)}")
        print(f"   New Datatype Props    : {len(new_data)}")

        # Filter out anything that already has a lesson
        filtered_new = []

        for u in sorted(new_classes):
            if u in lesson_uris:
                continue
            filtered_new.append({"type": "Class", "uri": u})

        for u in sorted(new_obj):
            if u in lesson_uris:
                continue
            filtered_new.append({"type": "ObjectProperty", "uri": u})

        for u in sorted(new_data):
            if u in lesson_uris:
                continue
            filtered_new.append({"type": "DatatypeProperty", "uri": u})

        # Overwrite new_concepts.jsonl with only the filtered NEW ones
        with NEW_CONCEPTS_FILE.open("w", encoding="utf-8") as f:
            for rec in filtered_new:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if filtered_new:
            print(f"Found {len(filtered_new)} NEW DR concepts (after filtering lessons).")
            print("Written to new_concepts.jsonl")
        else:
            print("No new concepts to learn today (after filtering lessons).")
            NEW_CONCEPTS_FILE.write_text("", encoding="utf-8")

        # Always update baseline to the CURRENT DR snapshot
        baseline = {
            "classes": sorted(curr_classes),
            "object_properties": sorted(curr_obj),
            "datatype_properties": sorted(curr_data),
        }
        with PREVIOUS_FILE.open("w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        print("Baseline updated in previous_entities.json")

    print("OntoTune detection step finished.")