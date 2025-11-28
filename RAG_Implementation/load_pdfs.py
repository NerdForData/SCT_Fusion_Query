"""
RAG Implementation - PDF Text Extraction and Chunking
------------------------------------------------------

Purpose:
    Extract text from binning research papers (PDFs) and split into
    manageable chunks for embedding generation and retrieval.

Workflow:
    1. Scan Papers/ directory for all PDF files
    2. Extract text from each PDF page using PyPDF2
    3. Split text into overlapping chunks (1200 chars, 200 overlap)
    4. Save chunks with metadata to rag_chunks.jsonl

Chunking Strategy:
    - Max chunk size: 1200 characters (optimal for embedding models)
    - Overlap: 200 characters (preserves context across chunk boundaries)
    - Whitespace normalization to remove formatting artifacts

Inputs:
    Papers/*.pdf - Research papers on binning (stored in Papers/ subdirectory)

Outputs:
    rag_chunks.jsonl - JSON Lines file with structure:
        {"id": "paper.pdf::chunk_0", "source": "paper.pdf", 
         "chunk_index": 0, "text": "..."}

Usage:
    python load_pdfs.py
    
Next Step:
    Run embed_chunks.py to generate embeddings for these chunks
"""

from pathlib import Path
import json
from PyPDF2 import PdfReader

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PDF_DIR = Path(__file__).parent / "Papers"                      # Input: directory containing PDF files
OUT_FILE = Path(__file__).parent.parent / "rag_chunks.jsonl"    # Output: chunked text with metadata

# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_text(pdf_path):
    """
    Extract all text content from a PDF file.
    
    Uses PyPDF2 to iterate through pages and extract text. Handles
    problematic pages gracefully by skipping them.
    
    Args:
        pdf_path (Path): Path to the PDF file
    
    Returns:
        str: Concatenated text from all successfully extracted pages
    
    Raises:
        Exception: If PDF cannot be opened or is completely unreadable
    """
    try:
        # Open PDF and initialize reader
        reader = PdfReader(str(pdf_path))
        text = []
        
        # Extract text from each page
        for page in reader.pages:
            try:
                # Extract text, handle None case
                t = page.extract_text() or ""
                text.append(t)
            except Exception as e:
                # Skip problematic pages (corrupted, image-only, etc.)
                continue
        
        # Concatenate all page text with newlines
        return "\n".join(text)
        
    except Exception as e:
        # Raise exception if entire PDF is unreadable
        raise Exception(f"Failed to read PDF: {str(e)[:100]}")

# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def split_into_chunks(text, max_len=1200, overlap=200):
    """
    Split long text into overlapping chunks for embedding generation.
    
    Chunking Strategy:
        - Normalizes whitespace to remove PDF formatting artifacts
        - Creates fixed-size chunks with overlap to preserve context
        - Overlap ensures important information at boundaries isn't lost
    
    Args:
        text (str): Raw text extracted from PDF
        max_len (int): Maximum characters per chunk (default: 1200)
        overlap (int): Character overlap between consecutive chunks (default: 200)
    
    Returns:
        list[str]: List of text chunks
    
    Example:
        text = "Long document text..." (3000 chars)
        chunks = split_into_chunks(text, max_len=1200, overlap=200)
        # Returns: [text[0:1200], text[1000:2200], text[2000:3000]]
    """
    # Normalize whitespace (collapse multiple spaces, remove newlines)
    text = " ".join(text.split())
    
    # If text is shorter than max_len, return as single chunk
    if len(text) <= max_len:
        return [text]

    chunks, start = [], 0
    
    # Create overlapping chunks using sliding window
    while start < len(text):
        # Calculate end position (don't exceed text length)
        end = min(start + max_len, len(text))
        
        # Extract chunk
        chunks.append(text[start:end])
        
        # Stop if we've reached the end
        if end >= len(text):
            break
        
        # Move start position back by overlap amount for next chunk
        # This creates overlap: next chunk starts (max_len - overlap) ahead
        start = end - overlap

    return chunks

# ---------------------------------------------------------
# MAIN PROCESSING PIPELINE
# ---------------------------------------------------------

def main():
    """
    Main function to process all PDFs and generate chunked output.
    
    Process:
        1. Delete existing output file (fresh start)
        2. Iterate through all PDFs in Papers/ directory
        3. Extract text from each PDF
        4. Split text into chunks with overlap
        5. Save chunks with metadata to JSONL file
        6. Handle errors gracefully (skip problematic PDFs)
    
    Output Format:
        Each line in rag_chunks.jsonl:
        {
            "id": "paper_name.pdf::chunk_0",     # Unique chunk identifier
            "source": "paper_name.pdf",          # Source PDF filename
            "chunk_index": 0,                    # Position in source document
            "text": "..."                        # Chunk text content
        }
    """
    # Remove existing output file (start fresh)
    if OUT_FILE.exists():
        OUT_FILE.unlink()

    # Open output file in append mode
    with OUT_FILE.open("a", encoding="utf-8") as fout:
        # Process each PDF file in Papers/ directory
        for pdf in PDF_DIR.glob("*.pdf"):
            try:
                # Extract text from PDF
                raw = extract_text(pdf)
                
                # Split into overlapping chunks
                chunks = split_into_chunks(raw)

                # Save each chunk with metadata
                for i, chunk in enumerate(chunks):
                    rec = {
                        "id": f"{pdf.name}::chunk_{i}",          # Unique ID
                        "source": pdf.name,                       # Source PDF
                        "chunk_index": i,                         # Position
                        "text": chunk                             # Content
                    }
                    # Write as JSON Line (one JSON object per line)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                print(f"[INFO] {pdf.name}: {len(chunks)} chunks")
                
            except Exception as e:
                # Skip problematic PDFs and continue processing
                print(f"[WARNING] Skipping {pdf.name}: {str(e)[:100]}")
                continue

    print("[SUCCESS] Done. Saved chunks to rag_chunks.jsonl")

# Execute main pipeline when script is run directly
if __name__ == "__main__":
    main()