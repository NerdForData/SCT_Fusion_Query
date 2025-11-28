# Semantic-Condition Transformation (SCT) with Conditional RAG Fusion

**Master Thesis Project - Improvement of Semiconductor Understanding using Semantic Web**  
**Author:** Aishwarya  
**Supervisor from Infineon:** Mr. Hans Ehm, Ms. Marta Bonik and Mr. Abdelgafar Ismail
**Supervisor from University:** Mr. Ruben Nuredini
---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Description](#modules-description)
- [Configuration](#configuration)
- [Workflow](#workflow)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements a **hybrid question-answering system** that combines:

1. **Semantic-Condition Transformation (SCT)**: Semantic search over Digital Reference ontology
2. **Conditional RAG Pipeline**: Context-aware retrieval from binning research papers
3. **LLM-based Fusion**: GPT-4o powered answer generation with multi-source context

### Key Features

- ✅ **Intelligent Intent Classification**: Automatically detects binning-related questions
- ✅ **Multi-Source Context Fusion**: Combines ontology knowledge + research papers + LLM reasoning
- ✅ **Auto-Update Detection**: Monitors Digital Reference changes and regenerates embeddings
- ✅ **Checkpoint/Resume Support**: Robust embedding generation with progress tracking
- ✅ **Rate Limiting & Error Handling**: Production-ready with retry logic and token management

### Use Cases

- Technical documentation Q&A for semiconductor manufacturing
- Binning yield optimization queries
- Digital Reference ontology exploration
- Research-backed answer generation for domain-specific questions

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Intent Classifier (intent_classifier.py)           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Regex-based detection for bin/bins/binned/binning terms  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────┬───────────────────────┘
                  │                       │
         Binning Question?        Non-Binning Question
                  │                       │
                  ▼                       ▼
┌─────────────────────────────┐  ┌──────────────────────┐
│  RAG Retrieval (ACTIVATED)  │  │  RAG Retrieval (OFF) │
│  rag_retriever.py           │  └──────────────────────┘
│  ┌────────────────────────┐ │             │
│  │ 1. Embed query vector  │ │             │
│  │ 2. FAISS search (top-5)│ │             │
│  │ 3. Return paper chunks │ │             │
│  └────────────────────────┘ │             │
└──────────────┬──────────────┘             │
               │                            │
               └────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         SCT Semantic Search (SCT_semantic_graph.py)             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Embed query using GPT4IFX (sfr-embedding-mistral)     │   │
│  │ 2. Search FAISS index (DR_faiss.index)                   │   │
│  │ 3. Retrieve top-3 ontology contexts                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Fusion & Answer Generation (GPT-4o)                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Prompt Construction:                                     │   │
│  │   • System Rules (Infineon guidelines)                   │   │
│  │   • Digital Reference Context (SCT)                      │   │
│  │   • Research Paper Context (RAG - if binning)            │   │
│  │   • User Question                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Answer + Context                     │
│  • Technical answer with follow-up questions                    │
│  • SCT context used                                             │
│  • RAG chunks (if applicable)                                   │
│  • Intent classification result                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Code/
├── System_KG_Implementation/          # Main SCT + Ontology System
│   ├── SCT_fusion_query.py            # Main (fusion) query engine
│   ├── SCT_semantic_graph.py          # Auto-update detection & embedding generation
│   ├── DR_load.py                     # Digital Reference TTL parser
│   ├── build_corpus.py                # Corpus builder from ontology
│   ├── build_faiss_index.py           # FAISS index generator
│   ├── extract_classes.py             # Extract ontology classes
│   ├── extract_properties.py          # Extract ontology properties
│   ├── base_file.py                   # Authentication & token testing utility
│   ├── DigitalReference.ttl           # Digital Reference ontology (TTL format)
│   ├── semantic_conditions.jsonl      # Pre-generated SCT embeddings
│   ├── DR_faiss.index                 # FAISS index for ontology
│   ├── binning_faiss.index            # FAISS index for binning papers
│   └── binning_docs.jsonl             # Document metadata for RAG
│
├── RAG_Implementation/                # Conditional RAG Pipeline
│   ├── load_pdfs.py                   # PDF extraction & chunking
│   ├── embed_chunks.py                # Embedding generation (checkpoint support)
│   ├── intent_classifier.py           # 🎯 Binning question detector
│   ├── rag_retriever.py               # FAISS-based semantic search
│   └── Papers/                        # Research papers (PDF format)
│
├── SCT_UI/                            # 🌐 Web User Interface
│   ├── app.py                         # Flask application server
│   ├── requirements.txt               # UI-specific dependencies
│   ├── templates/
│   │   └── index.html                 # Main chat interface (HTML + JavaScript)
│   └── static/
│       └── style.css                  # Modern dark theme styling
│
├── GPT_PromptSet/                     # Standalone GPT Client
│   ├── cli.py                         # Command-line interface
│   ├── gpt_client.py                  # GPT4IFX wrapper with auth
│   └── rules.txt                      # System prompt rules
│
├── venv/                              # Python virtual environment
└── README.md                          # 📖 This file
```

---

## 🛠️ Prerequisites

### Software Requirements

- **Python**: 3.12+ (tested with Python 3.12.6)
- **Operating System**: Windows 10/11 (PowerShell)
- **Network Access**: Infineon internal network (for GPT4IFX API)

### Python Packages

```txt
openai>=1.0.0
httpx
faiss-cpu (or faiss-gpu)
numpy
PyPDF2
rdflib
requests
urllib3
tqdm
Flask>=3.0.0          # For web UI
Werkzeug>=3.0.0       # Flask dependency
python-dotenv>=1.0.0  # Environment configuration
```

### API Access

- **GPT4IFX Endpoint**: `https://gpt4ifx.icp.infineon.com`
- **Credentials**: Infineon Active Directory account
- **Models Used**:
  - `gpt-4o` (LLM for answer generation)
  - `sfr-embedding-mistral` (4096-dimensional embeddings)

---

## 📦 Installation

### Step 1: Clone/Download Project

```powershell
cd C:\Users\Aishwarya\Desktop\Thesis\Code
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

```powershell
pip install openai httpx faiss-cpu numpy PyPDF2 rdflib requests urllib3 tqdm Flask Werkzeug python-dotenv
```

**Or install from requirements.txt (if available):**
```powershell
pip install -r requirements.txt
```

### Step 4: Verify Installation

```powershell
python -c "import openai, faiss, numpy; print('All packages installed successfully')"
```

---

## 🚀 Usage

### 1. SCT Fusion Query (Main System)

Run the main question-answering system with SCT + Conditional RAG:

```powershell
cd System_KG_Implementation
python SCT_fusion_query.py
```

**Interactive Mode:**
```
Enter your question (or 'quit' to exit): What is binning?
[INTENT] Intent classified: binning=True
[SCT] Retrieving Digital Reference context...
[RAG] Retrieving binning research papers...
[ANSWER] <Generated answer with follow-up questions>
```

### 2. Update SCT Embeddings (Auto-Detect Changes)

Automatically regenerate embeddings if Digital Reference is updated:

```powershell
python SCT_semantic_graph.py
```

**Features:**
- Hash-based change detection
- Resume from checkpoint if interrupted
- Rate limiting (15 requests/minute)

### 3. RAG Pipeline (Binning Papers)

#### Step 3a: Extract Text from PDFs

```powershell
cd RAG_Implementation
python load_pdfs.py
```

**Output:** `rag_chunks.jsonl` (text chunks with 1200 char max, 200 overlap)

#### Step 3b: Generate Embeddings

```powershell
python embed_chunks.py
```

**Output:** 
- `binning_faiss.index` (FAISS vector index)
- `binning_docs.jsonl` (document metadata)

**Features:**
- Checkpoint every 10 chunks
- Token refresh every 500 embeddings
- Auto-resume on interruption

### 4. Web User Interface

Launch the Flask-based web interface for interactive querying:

```powershell
cd SCT_UI
python app.py
```

**Access the UI:**
- Open browser to `http://localhost:5000`
- Modern chat interface with real-time responses
- Visual indicators for RAG/SCT activation
- Follow-up question suggestions
- Collapsible context details

**Features:**
- ✅ **Interactive Chat**: Clean, responsive interface with example questions
- ✅ **Real-time Processing**: Loading indicators and smooth animations
- ✅ **Context Visualization**: View SCT and RAG contexts used for each answer
- ✅ **Follow-up Questions**: AI-generated related questions for exploration
- ✅ **Conversation Management**: Reset functionality to start fresh
- ✅ **LaTeX Rendering**: Mathematical formulas displayed with KaTeX
- ✅ **Markdown Support**: Proper formatting for headings, lists, and emphasis

**UI Controls:**
- **Enable/Disable Follow-up Questions**: Toggle to get AI-suggested related questions
- **Reset Conversation**: Clear chat history and start over
- **Example Questions**: Quick-start buttons for common queries

### 5. Standalone GPT Client

Simple GPT-4o query with custom rules:

```powershell
cd GPT_PromptSet
python cli.py "What are the temperature specifications?"
```

**Or interactive mode:**
```powershell
python cli.py
Your question here please: <enter question>
```

---

## 📚 Modules Description

### System_KG_Implementation/

#### `SCT_fusion_query.py` 
**Main fusion query engine** that combines LLM + SCT + Conditional RAG.

**Key Functions:**
- `get_auth_token()`: Authenticate with GPT4IFX
- `retrieve_sct_context(query)`: Semantic search over Digital Reference
- `compress_sct(results)`: Format ontology context for prompt
- `ask_question(question)`: Main query pipeline
- `suggest_followup(question, answer)`: Generate follow-up questions

**Example:**
```python
answer, sct_ctx, rag_ctx, is_binning = ask_question("What is bin yield?")
```

#### `SCT_semantic_graph.py`
**Auto-update detection and embedding generation** for Digital Reference ontology.

**Features:**
- SHA-256 hash comparison for change detection
- Checkpoint/resume support
- Token refresh every 500 embeddings
- Generates `semantic_conditions.jsonl` + `DR_faiss.index`

#### `DR_load.py`
**Digital Reference ontology parser** (TTL/RDF format).

**Extracts:**
- Classes with labels and descriptions
- Object properties with domains/ranges
- Datatype properties with datatypes

#### `build_corpus.py`
**Corpus builder** that combines classes + properties into unified corpus.

**Output:** `DR_corpus.jsonl`

#### `build_faiss_index.py`
**FAISS index generator** for ontology embeddings.

**Output:** `DR_faiss.index` + `DR_ids.npy`

#### `base_file.py`
**Authentication test utility** for verifying GPT4IFX token generation.

### RAG_Implementation/

#### `intent_classifier.py` 
**Binning question detector** using regex-based word boundary matching.

**Detects:**
- bin, bins, binned, binning (with word boundaries)
- Compound terms: "bin yield", "test bin", "binning yield"

**Avoids false positives:** combine, binary, cabin

#### `load_pdfs.py`
**PDF text extraction and chunking** for research papers.

**Parameters:**
- Max chunk size: 1200 characters
- Overlap: 200 characters
- Whitespace normalization

#### `embed_chunks.py`
**Embedding generation** with checkpoint/resume support.

**Features:**
- Saves checkpoint every 10 chunks
- Token refresh every 500 embeddings
- Zero-vector fallback on error
- Rate limiting (~4.5s per request)

#### `rag_retriever.py`
**FAISS-based semantic search** for binning papers.

**Key Functions:**
- `embed_query(query)`: Generate + normalize query embedding
- `retrieve_binning_chunks(query, top_k=3)`: Return top-k chunks with scores

### SCT_UI/

#### `app.py`
**Flask web application** for interactive SCT Fusion Query interface.

**Key Features:**
- RESTful API endpoints for query processing
- Session-based conversation history
- Integration with `SCT_fusion_query.py` backend
- Error handling and logging

**API Endpoints:**
- `GET /`: Render main chat interface
- `POST /get-response`: Process user query and return answer
  - Request: `{"query": "...", "get_followup": true/false}`
  - Response: `{"answer": "...", "sct_context": "...", "rag_context": "...", "is_binning": true/false, "followup_questions": [...]}`
- `POST /reset`: Clear conversation history
- `GET /health`: Server health check

#### `templates/index.html`
**Modern chat interface** with responsive design.

**Features:**
- Real-time chat with user/bot message distinction
- Loading overlay with spinner animation
- Example question buttons for quick start
- Collapsible context details (SCT/RAG)
- Badge indicators for RAG/SCT activation
- Follow-up question suggestions with toggle
- Conversation reset functionality
- LaTeX formula rendering via KaTeX
- Markdown formatting (headings, bold, lists)

#### `static/style.css`
**Professional dark theme** with modern UI elements.

**Styling:**
- Gradient backgrounds and smooth animations
- Custom scrollbars and hover effects
- Responsive design for mobile devices
- Formatted text styling (headings, lists, emphasis)
- LaTeX formula highlighting
- Color-coded badges and context sections

### GPT_PromptSet/

#### `cli.py`
**Command-line interface** for standalone GPT queries.

**Supports:**
- Command-line arguments
- Interactive input mode
- Word count display

#### `gpt_client.py`
**GPT4IFX wrapper** with authentication and word limit enforcement.

**Features:**
- Bearer token authentication
- 200-word limit with auto-retry
- Temperature control (default: 0.4)

---

## ⚙️ Configuration

### API Settings

```python
BASE_URL = "https://gpt4ifx.icp.infineon.com"
MODEL_LLM = "gpt-4o"
MODEL_EMB = "sfr-embedding-mistral"
```

### Authentication

```python
USERNAME = "INFINEON\\Aishwarya"
PASSWORD = "Gauri@123456789"
```

### Retrieval Parameters

```python
TOP_K_SCT = 3        # Digital Reference contexts
TOP_K_RAG = 5        # Research paper chunks
TEMPERATURE = 0.15   # LLM temperature (low for factual answers)
```

### Rate Limiting

```python
MAX_CALLS_MIN = 15              # 15 requests per minute
SLEEP = 60/MAX_CALLS_MIN + 0.5  # ~4.5 seconds between requests
```

---

## 🔄 Workflow

### End-to-End Pipeline

1. **User submits question** → `SCT_fusion_query.py`
2. **Intent classification** → `intent_classifier.py` determines if binning-related
3. **SCT retrieval** → Semantic search over Digital Reference ontology
4. **Conditional RAG** → Fetch research papers only if binning question
5. **Prompt construction** → Combine rules + SCT + RAG contexts
6. **LLM generation** → GPT-4o generates answer with low temperature
7. **Return results** → Answer + contexts + intent flag

### Update Digital Reference

1. **Replace** `DigitalReference.ttl` with new version
2. **Run** `python SCT_semantic_graph.py` (auto-detects changes via hash)
3. **Generates** new `semantic_conditions.jsonl` and `DR_faiss.index`

### Add New Research Papers

1. **Place PDFs** in `RAG_Implementation/Papers/`
2. **Extract text**: `python load_pdfs.py`
3. **Generate embeddings**: `python embed_chunks.py`
4. **Files created**: `binning_faiss.index`, `binning_docs.jsonl`

---

## 📖 API Documentation

### `ask_question(question: str) → tuple`

Main query function in `SCT_fusion_query.py`.

**Parameters:**
- `question` (str): User's natural language question

**Returns:**
- `answer` (str): Generated answer
- `sct_context` (str): Digital Reference contexts used
- `rag_context` (str): Research paper chunks used (empty if not binning)
- `is_binning` (bool): Intent classification result

**Example:**
```python
answer, sct, rag, is_bin = ask_question("How are chips binned?")
print(f"Answer: {answer}")
print(f"Used RAG: {is_bin}")
```

### `is_binning_question(question: str) → bool`

Intent classifier in `intent_classifier.py`.

**Parameters:**
- `question` (str): User's question

**Returns:**
- `bool`: True if binning-related, False otherwise

**Example:**
```python
from intent_classifier import is_binning_question

is_bin = is_binning_question("What is bin yield?")  # True
is_bin = is_binning_question("What is temperature?")  # False
```

### `retrieve_binning_chunks(query: str, top_k: int = 3) → list`

RAG retriever in `rag_retriever.py`.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of chunks to return (default: 3)

**Returns:**
- `list[dict]`: Ranked chunks with structure:
  ```python
  {
      "source": str,    # PDF filename
      "score": float,   # Similarity score (0-1)
      "text": str       # Chunk text (max 600 chars)
  }
  ```

**Example:**
```python
from rag_retriever import retrieve_binning_chunks

chunks = retrieve_binning_chunks("bin yield optimization", top_k=3)
for chunk in chunks:
    print(f"{chunk['source']}: {chunk['score']:.2f}")
```

---

## 🐛 Troubleshooting

### Issue: `FileNotFoundError: ca-bundle.crt`

**Solution:** SSL verification is disabled for internal network. Update code:
```python
http_client=httpx.Client(verify=False)  # Instead of verify=str(CERT_PATH)
```

### Issue: `ModuleNotFoundError: No module named 'intent_classifier'`

**Solution:** Check path configuration in `SCT_fusion_query.py`:
```python
rag_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RAG_Implementation")
```

### Issue: Intent classifier returns `False` for binning questions

**Solution:** Restart Python session to clear module cache:
```powershell
# Exit Python and restart
python SCT_fusion_query.py
```

### Issue: `RuntimeError: FAISS index not found`

**Solution:** Check file paths and ensure indexes are generated:
```powershell
# For ontology index
python build_faiss_index.py

# For RAG index
cd RAG_Implementation
python embed_chunks.py
```

### Issue: Authentication fails

**Solution:** Verify credentials and network access:
```powershell
python base_file.py  # Test authentication
```

### Issue: Rate limit exceeded

**Solution:** Adjust sleep time in configuration:
```python
SLEEP = 5.0  # Increase to 5 seconds between requests
```

---

## 🤝 Contributing

### Code Style

- Follow existing comment structure with detailed docstrings
- Use type hints where applicable
- Keep functions focused and well-documented

### Adding New Features

1. Create feature branch
2. Add comprehensive comments
3. Test with multiple edge cases
4. Update README.md

### Testing

```powershell
# Test intent classifier
python test_intent.py

# Test authentication
python base_file.py

# Test RAG retrieval
python -c "from rag_retriever import retrieve_binning_chunks; print(retrieve_binning_chunks('test'))"
```

---

## 📝 License

**Proprietary - Infineon Technologies**  
For internal research and development use only.

---

## 📧 Contact

**Author:** Aishwarya  
**Organization:** Infineon Technologies  
**Project Type:** Master Thesis  

For questions or issues, please contact the Infineon Research Team.

---

## 🎓 Acknowledgments

- **Infineon Technologies** for providing infrastructure and API access
- **GPT4IFX Team** for internal LLM deployment
- **Thesis Supervisor** for guidance and support

---

**Last Updated:** November 2025  
**Version:** 1.0.0
