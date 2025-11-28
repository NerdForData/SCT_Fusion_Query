"""
SCT + Conditional RAG Fusion Query
----------------------------------

This pipeline combines:
- LLM internal knowledge
- Digital Reference SCT semantic vectors
- Conditional RAG (only for binning-related questions)
- Infineon answer rules
- Optional follow-up question suggestions
"""

import json, numpy as np, faiss, openai, httpx, time, base64, requests, urllib3 
from tqdm import tqdm
import sys
import os

# ------------------------------------------------------------------
# IMPORT RAG MODULES  
# ------------------------------------------------------------------
# Dynamically add RAG_Implementation folder to Python path to import custom modules
rag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RAG_Implementation")

if rag_path not in sys.path:
    sys.path.insert(0, rag_path)

try:
    # Import intent classifier to detect binning-related questions
    from intent_classifier import is_binning_question
    # Import RAG retriever to fetch relevant research paper chunks
    from rag_retriever import retrieve_binning_chunks
except Exception as e:
    print(f"[ERROR] Failed to import RAG modules from: {rag_path}")
    print(f"Error: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Disable SSL warnings for internal Infineon network
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Infineon internal GPT4IFX API endpoint
BASE_URL  = "https://gpt4ifx.icp.infineon.com"
MODEL_LLM = "gpt-4o"                    # LLM model for answer generation
MODEL_EMB = "sfr-embedding-mistral"     # Embedding model for semantic search

# Retrieval parameters
TOP_K_SCT = 3               # Number of Digital Reference semantic context entries to retrieve
TOP_K_RAG = 5               # Number of research paper chunks to retrieve for binning questions
TEMPERATURE = 0.15          # Low temperature for consistent, factual answers

# ------------------------------------------------------------------
# AUTHENTICATION
# ------------------------------------------------------------------
def get_auth_token():
    """
    Authenticate with GPT4IFX and retrieve Bearer token.
    Uses Basic Auth to exchange credentials for a session token.
    
    Returns:
        str: Bearer token for API authentication
    """
    username = "INFINEON\\Aishwarya"
    password = "Gauri@123456789"

    # Encode credentials as Base64 for Basic Auth
    basic_token = base64.b64encode(f"{username}:{password}".encode("ascii")).decode("ascii")
    headers = {"Authorization": f"Basic {basic_token}", "Content-Type": "application/json"}

    # Request Bearer token from authentication endpoint
    resp = requests.get(f"{BASE_URL}/auth/token", headers=headers, verify=False, auth=(username, password))

    if resp.status_code == 200:
        # Extract token from custom header
        token = resp.headers.get("x-forwarded-access-token")
        if token:
            return token
    raise Exception(f"Authentication failed ({resp.status_code})")

print("[AUTH] Getting authentication token...")
TOKEN = get_auth_token()
print("[SUCCESS] Auth OK")

client = openai.OpenAI(
    api_key=TOKEN,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=False)
)

# ------------------------------------------------------------------
# LOAD SEMANTIC CONDITIONS (SCT vectors)
# ------------------------------------------------------------------
# Load pre-generated Digital Reference semantic embeddings for context retrieval
print("[LOAD] Loading semantic_conditions.jsonl...")

# Initialize storage for labels, context texts, and embedding vectors
labels, contexts, vectors = [], [], []
semantic_file = os.path.join(os.path.dirname(__file__), "semantic_conditions.jsonl")
with open(semantic_file, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            labels.append(obj["label"])              # Ontology concept label
            contexts.append(obj["context_text"])     # Human-readable context
            vectors.append(np.array(obj["embedding"], dtype="float32"))  # Embedding vector

# Stack all vectors into a matrix for FAISS indexing
emb_matrix = np.vstack(vectors)
dim = emb_matrix.shape[1]
faiss.normalize_L2(emb_matrix)  # Normalize for cosine similarity (Inner Product)

# Create FAISS index for fast nearest neighbor search
index = faiss.IndexFlatIP(dim)  # Inner Product index for normalized vectors
index.add(emb_matrix)
print(f"[LOAD] Loaded {len(labels)} SCT vectors.")

# ------------------------------------------------------------------
# SCT CONTEXT RETRIEVAL
# ------------------------------------------------------------------
def retrieve_sct_context(question, k=TOP_K_SCT):
    """
    Retrieve the most semantically similar Digital Reference contexts for a given question.
    
    Args:
        question (str): User's input question
        k (int): Number of top similar contexts to retrieve
    
    Returns:
        list: List of dicts with 'label', 'context', and 'score' keys
    """
    # Generate embedding for the user's question
    emb = client.embeddings.create(
        model=MODEL_EMB,
        input=[question],
        encoding_format="float"
    ).data[0].embedding

    # Convert to numpy array and normalize for cosine similarity
    qv = np.array([emb], dtype="float32")
    faiss.normalize_L2(qv)
    
    # Search FAISS index for k nearest neighbors
    D, I = index.search(qv, k)  # D = distances (scores), I = indices

    # Return retrieved contexts with their labels and similarity scores
    return [
        {"label": labels[idx], "context": contexts[idx], "score": float(Di)}
        for idx, Di in zip(I[0], D[0])
    ]

def compress_sct(retrieved, max_chars=1200):
    """
    Compress retrieved SCT contexts to fit within token limits.
    Removes redundant subClassOf statements and limits context length.
    
    Args:
        retrieved (list): List of retrieved context dicts
        max_chars (int): Maximum character limit for compressed output
    
    Returns:
        str: Compressed context text
    """
    merged = []
    for r in retrieved:
        # Split context by delimiter and filter out subClassOf statements
        parts = [p for p in r["context"].split(" | ") if "subClassOf" not in p][:10]
        # Format with label header
        merged.append(f"[{r['label']}]\n" + " | ".join(parts))
    
    full = "\n\n".join(merged)
    return full[:max_chars]  # Truncate to max length


# ------------------------------------------------------------------
# INFINEON RULES
# ------------------------------------------------------------------
RULES = """
You are an Infineon semiconductor domain assistant.

General expectations:
1. Start with a direct, one-sentence definition or conclusion that answers the question.
2. Structure every answer as: Definition → Short Explanation → Optional Supporting Detail.
3. Keep total length within ~200 words (≈3–6 sentences).
4. Provide clear cause–effect or process reasoning rather than listing entities.
5. Use precise semiconductor terminology (wafer, lithography, GOX, linewidth, binning, yield).
6. Mention formulas or standards (Cp, Cpk, ISO, SPC) only if relevant and factual.
7. Keep discussion tied to semiconductor manufacturing, yield, or process control.
8. Write in clear, professional technical language; use short paragraphs or bullet points when needed.
9. Maintain a neutral, factual tone—no speculation or narrative phrasing.
10. Add an “Additional Notes:” section only if genuinely valuable (e.g., limitation, brief example).

Always combine your own engineering knowledge with the Digital Reference context to produce clear, concise, and technically sound answers.

If binning applies, integrate RAG engineering text directly.
"""


# ------------------------------------------------------------------
# FUSION REASONING (SCT + Conditional RAG)
# ------------------------------------------------------------------
def fused_reasoning(question):
    """
    Main reasoning pipeline combining multiple knowledge sources:
    1. Classify question intent (binning vs general)
    2. Retrieve Digital Reference semantic context (SCT)
    3. Conditionally retrieve research paper chunks (RAG) for binning questions
    4. Generate answer using LLM with combined context
    
    Args:
        question (str): User's input question
    
    Returns:
        tuple: (answer, sct_context, rag_context, is_binning_flag)
    """
    # Step 1 — Classify question intent
    # Determine if question is about binning/yield to decide if RAG is needed
    is_bin = is_binning_question(question)
    print(f"[INTENT] Intent classified: binning={is_bin}")

    # Step 2 — Retrieve Digital Reference semantic context
    # Get relevant ontology concepts and their contexts using semantic search
    sct = retrieve_sct_context(question)
    sct_text = compress_sct(sct)

    # Step 3 — Conditionally retrieve RAG context
    # Only fetch research paper chunks if question is binning-related
    rag_text = ""
    if is_bin:
        chunks = retrieve_binning_chunks(question, TOP_K_RAG)
        rag_text = "\n".join([f"- {c['text']}" for c in chunks])

    # Step 4 — Build final prompt with all contexts
    # Combine rules, SCT context, optional RAG context, and user question
    prompt = f"""
{RULES}

Digital Reference Context:
{sct_text}

{"Relevant Binning Research (RAG):\n" + rag_text if rag_text else ""}

Question: {question}

Provide a clear, technically correct, concise answer.
"""

    # Generate answer using LLM with low temperature for consistency
    resp = client.chat.completions.create(
        model=MODEL_LLM,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content.strip(), sct_text, rag_text, is_bin


# ------------------------------------------------------------------
# FOLLOW-UP QUESTION GENERATOR
# ------------------------------------------------------------------
def generate_followups(question, answer):
    """
    Generate follow-up questions based on the original Q&A to encourage exploration.
    Uses slightly higher temperature for more diverse suggestions.
    
    Args:
        question (str): Original user question
        answer (str): Generated answer
    
    Returns:
        str: Three numbered follow-up questions
    """
    prompt = f"""
Generate exactly 3 short follow-up semiconductor questions.

Original question: {question}
Answer: {answer}

Return only:
1. ...
2. ...
3. ...
"""

    # Use slightly higher temperature (0.3) for more creative follow-up suggestions
    resp = client.chat.completions.create(
        model=MODEL_LLM,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------
# INTERACTIVE SHELL
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n[READY] SCT + Conditional RAG Fusion Ready\n")

    # Main interactive loop
    while True:
        # Get user input (empty input exits)
        q = input("[INPUT] Your question: ").strip()
        if not q:
            break

        # Process question through fusion pipeline
        ans, sct_used, rag_used, bin_flag = fused_reasoning(q)

        # Display answer
        print("\n[ANSWER] Answer:")
        print(ans)

        # Generate and display follow-up suggestions
        print("\n------------------------------------------")
        print("[FOLLOWUP] Follow-up suggestions:")
        print(generate_followups(q, ans))
        print("------------------------------------------\n")

        # Brief pause before next question
        time.sleep(1)