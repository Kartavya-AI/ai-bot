import PyPDF2
from pinecone import Pinecone, ServerlessSpec
import re
import os
from dotenv import load_dotenv

# ========== CONFIG ==========
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
PDF_PATH = "Kartavya.pdf"  # Path to your PDF file

pc = Pinecone(api_key=PINECONE_API_KEY)
# ============================

# Step 1: Extract text from PDF (chunked)
def extract_text_chunks(pdf_path, max_chars=500):
    chunks = []
    buffer = ""

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue

            # Split into paragraphs or sentences using regex
            paragraphs = re.split(r"\n{2,}|\.\s", text)

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Add to buffer until it reaches max_chars
                if len(buffer) + len(para) <= max_chars:
                    buffer += para + " "
                else:
                    chunks.append(buffer.strip())
                    buffer = para + " "

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks
# Step 4: Initialize Pinecone
def init_pinecone():
    if not pc.has_index(INDEX_NAME):
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1", 
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
    
    # Return the index object
    return pc.Index(INDEX_NAME)

# Step 3: Upsert to Pinecone using built-in embeddings
def upsert_embeddings(chunks, index):
    # Prepare records for upsert_records method
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "_id": f"chunk-{i}",
            "text": chunk  # This field will be auto-embedded
        })
    
    try:
        # Use upsert_records for built-in embeddings (not upsert)
        # Using '__default__' for the default namespace
        index.upsert_records("__default__", records)
        print(f"[✓] Successfully upserted {len(records)} chunks using built-in embeddings")
    except Exception as e:
        print(f"[x] Batch upsert failed: {e}")
        # Fallback to individual upserts
        for i, chunk in enumerate(chunks):
            try:
                index.upsert_records("__default__", [{
                    "_id": f"chunk-{i}",
                    "chunk_text": chunk
                }])
                print(f"[✓] Upserted chunk-{i}")
            except Exception as e:
                print(f"[x] Error with chunk-{i}: {e}")

# MAIN
if __name__ == "__main__":
    print("[+] Extracting text from PDF...")
    text_chunks = extract_text_chunks(PDF_PATH)
    
    print("[+] Initializing Pinecone...")
    index = init_pinecone()
    
    print(f"[+] Uploading {len(text_chunks)} chunks to Pinecone...")
    upsert_embeddings(text_chunks, index)
    
    print("[✅] All chunks processed.")